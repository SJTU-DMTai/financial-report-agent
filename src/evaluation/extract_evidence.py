import re
import sys
import traceback
from json import JSONDecodeError
from pathlib import Path

import asyncio
import json
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from agentscope.model import ChatModelBase

from pipelines.planning import process_pdf_to_outline
from src.utils.instance import create_chat_model, create_agent_formatter
from src.utils.file_converter import pdf_to_markdown, markdown_to_sections
from src.memory.working import Section
from src.prompt import prompt_dict
from utils.call_with_retry import call_chatbot_with_retry
from utils.local_file import DEMO_DIR
from src.utils.instance import llm_reasoning, llm_instruct, formatter


def extract_text_from_content(content) -> str:
    """无论content是列表还是字典，都能安全地提取'text'值。"""
    if isinstance(content, list) and content:
        first_item = content[0]
        if isinstance(first_item, dict): return first_item.get('text', '').strip()
    elif isinstance(content, dict): return content.get('text', '').strip()
    return ""

async def get_content_from_response(response_msg) -> str:
    """统一处理流式和非流式的、各种格式的大模型响应。"""
    if hasattr(response_msg, 'content') and response_msg.content:
        return extract_text_from_content(response_msg.content)
    full_content = ""
    try:
        async for chunk in response_msg:
            if hasattr(chunk, 'content') and chunk.content:
                full_content += extract_text_from_content(chunk.content)
    except TypeError: pass
    return full_content.strip()

def get_all_evidences_from_section(section: Section) -> List[str]:
    """递归地从Section对象中收集所有论据文本。"""
    evidences = []
    if section.segments:
        for segment in section.segments:
            if segment.evidences:
                evidences.extend([e for e in segment.evidences if e and e.strip()])
    if section.subsections:
        for subsection in section.subsections:
            evidences.extend(get_all_evidences_from_section(subsection))
    return evidences

async def extract_unique_evidences_from_pdf(pdf_path: Path, save_dir: Path) -> List[str]:
    pdf_stem = pdf_path.stem  # 去掉.pdf后缀
    evidence_filename = f"{pdf_stem}_evidences.json"
    evidence_path = save_dir / evidence_filename
    # 尝试直接读取
    if evidence_path.exists():
        print(f"    - 检测到已有的evidences，加载: {evidence_filename}")
        evidences = json.loads(evidence_path.read_text(encoding="utf-8"))
        return evidences

    print(f"\n-> 开始提取并清洗文件: {pdf_path.name}")
    manuscript = await process_pdf_to_outline(pdf_path, save_dir, llm_reasoning, llm_instruct, formatter)
    print(f"  - 从 {pdf_path.name} 的结构中提取论据...")
    evidences = get_all_evidences_from_section(manuscript)
    print(f"  - 对 {len(evidences)} 条原始论据进行语义去重...")

    evidences = await drop_duplicate_evidences(evidences)
    # 保存到long_term_dir
    evidence_path.write_text(
        json.dumps(evidences, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"    -> Evidence已保存到: {evidence_path}")
    return evidences

async def drop_duplicate_evidences(evidences: List[str]) -> List[str]:
    sys_prompt = f"""你是一名专业的金融分析助手，擅长数据清洗和信息摘要。
你的任务是对以下从一份研报大纲中提取的“论据（evidences）”列表进行去重。列表中的许多条目可能措辞不同，但表达的是相同或相似的含义。每行是带有id的一条论据。

### 流程
遍历每一个序号对应的论据，然后检查所有论据，找到所有含义相同的论据，该组论据对应的一或多个id用空格隔开，单独放在一行。然后遍历下一个论据，如果该论据已在上文中被划分到某组，则跳过。最后输出所有组，每组一行，只包含数字id和空格。

### 要求
1. 完整遍历所有论据，确保没有遗漏。
2. 忽视论据提及的数据来源，忽视每条论据涉及“查询”等动词，请关注核心实体或指标。如果有时间范围约束，则不同时间的实体属于不同的论据，不应去重。
3. 如果某条论据本身就是独特的，没有与之重复的项，那么必须将它包含在最终的列表中。
#### 输出示例（不包含其他字符）
1 3 6
2 6 19
"""
    _evidences = [e.replace("查询", "").replace("确认", "").replace("计算", "")
                  .replace("获取", "") for e in evidences]
    evidences = []
    for e in _evidences:
        if e not in evidences:
            evidences.append(e)
    numbered_evidences = "\n".join(f"{i+1} {e}" for i, e in enumerate(evidences))
    def _parse(response: str) -> List[str]:
        groups = []
        seen_ids = set()
        for line in response.strip().splitlines():
            ids = line.strip().split()
            for _id in ids: assert re.match(r"^\d+$", _id), f"输出格式错误：输出了数字id之外的其他字符: {_id}"
            unique_ids = [id_ for id_ in ids if id_ not in seen_ids]
            if unique_ids:
                groups.append(unique_ids)
                seen_ids.update(unique_ids)
        unique_evidences = []
        for group in groups:
            first_id = int(group[0]) - 1
            unique_evidences.append(list(evidences)[first_id])
        return unique_evidences
    return await call_chatbot_with_retry(
        llm_instruct, formatter,
        sys_prompt, f"需要处理的论据列表如下：\n---\n{numbered_evidences}\n---",
        hook=_parse, handle_hook_exceptions=(AssertionError, TypeError), max_retries=3,
    )

def find_report_pairs(pdf_directory: Path, earliest_date: str = "2025-01-01", latest_date: str = "2025-12-31") -> List[Tuple[str, Path, Path]]:
    """
    配对时间相近的两个研报。

    Args:
        pdf_directory: PDF文件所在目录
        earliest_date: 最早日期，格式为 'YYYY-MM-DD'
        latest_date: 最晚日期，格式为 'YYYY-MM-DD'

    Returns:
        研报对列表，每项为 (stock_code, old_report_path, new_report_path)
    """
    print(f"--- 步骤 1: 正在扫描文件夹并配对研报 (时间范围: {earliest_date} 到 {latest_date}) ---")

    from datetime import datetime
    earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")

    # 按股票代码和报告日期存储所有研报
    stock_reports = defaultdict(list)

    for pdf_file in pdf_directory.glob("*.pdf"):
        parts = pdf_file.name.split("_")
        if len(parts) < 2: continue
        stock_code = parts[0]
        date_str = parts[1]

        if not stock_code.isdigit() or len(stock_code) < 6: continue

        # 解析日期
        report_date = datetime.strptime(date_str, "%Y-%m-%d")

        # 检查是否在指定时间范围内
        if not (earliest_dt <= report_date <= latest_dt): continue

        stock_reports[stock_code].append((report_date, pdf_file))

    pairs = []
    for stock_code, reports_list in stock_reports.items():
        if len(reports_list) < 2: continue

        # 按日期排序
        reports_list.sort(key=lambda x: x[0])

        # 配对相邻的两个报告
        for i in range(len(reports_list) - 1):
            old_date, old_path = reports_list[i]
            new_date, new_path = reports_list[i + 1]
            if old_date == new_date: continue
            pairs.append((stock_code, old_path, new_path))
            print(f"  - 成功配对: {stock_code} -> {old_path.name} vs {new_path.name}")

    print(f"配对完成！共找到 {len(pairs)} 对符合条件的研报。")
    return pairs

async def find_locations_in_outline(outline_content_str: str, evidences_to_find: List[str]) -> Dict[str, str]:
    print(f"  - 正在大纲中定位 {len(evidences_to_find)} 条共通论据的位置...")
    if not evidences_to_find:
        return {}
    
    numbered_evidences = "\n".join(f"EV_{i+1}: {e}" for i, e in enumerate(evidences_to_find))
    
    # 为了防止Prompt过长，对outline_content进行简化，只保留关键信息
    outline_json = json.loads(outline_content_str)
    simplified_outline = []
    def simplify_section(section, prefix):
        current_loc = f"{prefix}s{section.get('section_id')}"
        simplified_outline.append(f"章节点: {current_loc}, 标题: \"{section.get('title')}\"")
        if section.get('segments'):
            for i, seg in enumerate(section['segments']):
                seg_loc = f"{current_loc}_p{seg.get('segment_id', i+1)}"
                if seg.get('evidences'):
                    simplified_outline.append(f"  - 段落点: {seg_loc}, 论据: {json.dumps(seg.get('evidences'), ensure_ascii=False)}")
        if section.get('subsections'):
            for sub in section['subsections']:
                simplify_section(sub, f"{current_loc}_")
    simplify_section(outline_json, "")
    simplified_outline_text = "\n".join(simplified_outline)


    prompt = f"""
你是一名精准的文本定位专家。你的任务是在给定的研报大纲结构中，为一系列指定的论据找到它们的确切位置路径。

**背景与定义:**
1.  **研报大纲**: 以下是一份简化版的研报结构，每一行代表一个章节或段落。
2.  **位置路径格式**: 路径由 `s{id}` (章节) 和 `p{id}` (段落) 组成，层层相连。
    - `s{id}`: 代表 `section_id`。
    - `p{id}`: 代表 `segment_id`。
    - **示例**: `s1_s3_p2` 表示在 `section_id=1` 的章节下的 `section_id=3` 的子章节中的 `segment_id=2` 的段落。

**任务指令:**
1.  仔细阅读下面提供的 "研报大纲结构"。
2.  对于 "待查找论据列表" 中的每一条论据（以 EV_ID 开头），在大纲的 `evidences` 数组中找到与之**完全匹配或语义上非常相似**的条目。
3.  记录下该条目所属的 "段落点" (例如 `s1_s3_p2`)。
4.  你的输出必须是一个合法的JSON对象。对象的键（key）是论据的ID（如 "EV_1"），值（value）是你找到的位置路径字符串（如 "s1_s3_p2"）。

**重要规则:**
-   如果某条论据在大纲中找不到，请在JSON对象中将其值设为 "not_found"。
-   你的回答中，**绝对不能包含**除了这个JSON对象之外的任何其他文字、解释或注释。

**研报大纲结构:**
---
{simplified_outline_text}
---

**待查找论据列表:**
---
{numbered_evidences}
---

现在，请严格按照上述格式，返回包含所有论据位置的JSON对象。
"""
    try:
        response_msg = await llm_reasoning([{"role": "user", "content": prompt}])
        json_string = await get_content_from_response(response_msg)
        if not json_string: return {}
        if json_string.strip().startswith("```json"): json_string = json_string.strip()[7:-3]
        
        locations_map_by_id = json.loads(json_string)
        
        locations_map_by_text = {}
        for i, evidence in enumerate(evidences_to_find):
            ev_id = f"EV_{i+1}"
            location = locations_map_by_id.get(ev_id, "not_found_in_llm_response")
            locations_map_by_text[evidence] = location
            
        print("    -> 定位完成。")
        return locations_map_by_text
    except Exception as e:
        print(f"    ! 论据定位时发生严重错误: {e}。")
        return {e: "error_during_loc_finding" for e in evidences_to_find}

async def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    long_term_dir = PROJECT_ROOT / "data" / "memory" / "long_term"
    
    # 配对研报，指定时间范围
    report_pairs = find_report_pairs(DEMO_DIR, earliest_date="2025-01-01", latest_date="2025-12-31")
    if not report_pairs: print("未找到任何可处理的研报对，程序退出。"); return
    
    existing_results = []
    output_json_path = PROJECT_ROOT / "data" / "output" / "comparison_results.json"
    output_txt_path = PROJECT_ROOT / "data"  / "output" / "comparison_results.txt"
    if output_json_path.exists():
        with open(output_json_path, 'r', encoding='utf-8') as f:
            try: existing_results = json.load(f)
            except json.JSONDecodeError: print(f"警告: {output_json_path} 文件为空或已损坏，将创建新文件。")
    results_cache = {res['stock_code']: res for res in existing_results}
    print(f"检测到 {len(results_cache)} 个已保存的结果。")

    print("\n--- 步骤 2: 开始批量处理研报对... ---")
    for stock_code, old_path, new_path in report_pairs:
        try:
            print(f"\n======= 正在处理股票: {stock_code} =======")
            should_extract_and_compare = True
            cached_data = results_cache.get(stock_code)

            if cached_data and "common_evidences" in cached_data:
                first_common = cached_data["common_evidences"][0] if cached_data["common_evidences"] else {}
                if isinstance(first_common, dict) and "location_old" in first_common:
                    print(f"--- 股票 {stock_code} 已被完整处理（包括定位），完全跳过 ---")
                    continue

            evidences_old = await extract_unique_evidences_from_pdf(old_path, long_term_dir / "demonstration")
            evidences_new = await extract_unique_evidences_from_pdf(new_path, long_term_dir / "demonstration")

            common_evidences_texts = await drop_duplicate_evidences(evidences_old + evidences_new)

            common_evidences_with_locs = []
            if common_evidences_texts:
                print(f"\n  --- 开始为 {len(common_evidences_texts)} 条共通论据定位 ---")
                outline_old_path = long_term_dir / "demonstration" / f"{old_path.stem}_outline.json"
                outline_new_path = long_term_dir / "demonstration" / f"{new_path.stem}_outline.json"

                if outline_old_path.exists() and outline_new_path.exists():
                    outline_old_content = outline_old_path.read_text(encoding='utf-8')
                    outline_new_content = outline_new_path.read_text(encoding='utf-8')

                    locations_old = await find_locations_in_outline(outline_old_content, common_evidences_texts)
                    locations_new = await find_locations_in_outline(outline_new_content, common_evidences_texts)

                    for text in common_evidences_texts:
                        common_evidences_with_locs.append({
                            "text": text,
                            "location_old": locations_old.get(text, "not_found"),
                            "location_new": locations_new.get(text, "not_found")
                        })
                else:
                    print("    ! 缺少outline.json文件，无法进行定位。")
                    common_evidences_with_locs = [{"text": text, "location_old": "outline_missing", "location_new": "outline_missing"} for text in common_evidences_texts]

            # --- [新逻辑] 开始：过滤掉任何包含 "not_found" 的共通论据 ---

            initial_count = len(common_evidences_with_locs)

            filtered_common_evidences = [
                item for item in common_evidences_with_locs
                if "not_found" not in item.get("location_old", "not_found") and \
                   "not_found" not in item.get("location_new", "not_found")
            ]

            filtered_count = len(filtered_common_evidences)
            if initial_count > filtered_count:
                print(f"  - 过滤完成：移除了 {initial_count - filtered_count} 条无法在两份报告中同时定位的共通论据。")

            # --- [新逻辑] 结束 ---

            result_data = {
                "stock_code": stock_code, "old_report": old_path.name, "new_report": new_path.name,
                "old_evidence_count": len(evidences_old), "new_evidence_count": len(evidences_new),
                "common_evidence_count": filtered_count, # 使用过滤后的数量
                "common_evidences": filtered_common_evidences # 保存过滤后的列表
            }

            results_cache[stock_code] = result_data
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(list(results_cache.values()), f, ensure_ascii=False, indent=4)
            print(f"--- 股票 {stock_code} 处理完成，结果已更新到 {output_json_path.name} ---")
        except Exception as e:
            traceback.print_exc()
            print(f"!!! 处理股票 {stock_code} 时发生错误: {e}，跳过该股票。")
            continue
    print("\n--- 步骤 3: 所有股票处理完毕，正在生成最终排序报告... ---")
    if not results_cache: print("未找到任何结果，无法生成排序报告。"); return
    
    final_results = list(results_cache.values())
    final_results.sort(key=lambda x: x["common_evidence_count"], reverse=True)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("研报共通论据数量排序 \n====================================\n\n")
        for i, res in enumerate(final_results):
            f.write(f"第 {i+1} 名: 股票代码 {res['stock_code']}\n")
            f.write(f"  - 共通论据数量: {res['common_evidence_count']}\n")
            f.write(f"  - (old论据数: {res['old_evidence_count']}, new论据数: {res['new_evidence_count']})\n")
            f.write(f"  - old 报告: {res['old_report']}\n")
            f.write(f"  - new 报告: {res['new_report']}\n\n")

    print(f"排序报告已生成: {output_txt_path}"); print("--- 全部任务完成！ ---")


if __name__ == "__main__":
    asyncio.run(main())