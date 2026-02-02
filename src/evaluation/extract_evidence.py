import re
import traceback
from pathlib import Path

import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from filelock import FileLock

from pipelines.planning import process_pdf_to_outline
from src.memory.working import Section
from utils.call_with_retry import call_chatbot_with_retry
from utils.local_file import DEMO_DIR
from src.utils.instance import llm_reasoning, llm_instruct, formatter

# 最大并发数限制
MAX_CONCURRENT = 4


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

    print(f"\n-> 开始提取并清洗文件: {pdf_path.name}", flush=True)
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
遍历每一个序号对应的论据，然后检查所有论据，找到所有含义相同的论据，该组论据对应的一或多个id用空格隔开，单独放在一行。然后遍历下一个论据，如果该论据已在上文中被划分到某组，则跳过。
最后输出所有组用<ANSWER>和</ANSWER>包裹住，每组一行，只包含数字id和空格。

""" + """
### 要求
1. 完整遍历所有论据，确保没有遗漏。
2. 忽视论据提及的数据来源，忽视每条论据涉及“查询”等动词，请关注核心实体或指标。如果有时间范围约束，则不同时间的实体属于不同的论据，不应去重。
3. 如果某条论据本身就是独特的，没有与之重复的项，那么必须将它包含在最终的列表中。
""" if len(evidences) < 300 else f"""
### 要求
1. 遍历每条论据。
2. 忽视论据提及的数据来源，忽视每条论据涉及“查询”等动词，请关注核心实体或指标。如果有时间范围约束，则不同时间的实体属于不同的论据，不应去重。
3. 如果某条论据本身就是独特的，没有与之重复的项，那么跳过，不必输出该论据id。（即每一组论据至少有两个不同的id）
""" + """
#### 输出示例
<ANSWER>
1 3 6
2 6 19
</ANSWER>
"""
    _evidences = [e.replace("查询", "").replace("确认", "").replace("计算", "")
                  .replace("获取", "") for e in evidences]
    evidences = []
    for e in _evidences:
        if e not in evidences:
            evidences.append(e)
    numbered_evidences = "\n".join(f"{i+1} {e}" for i, e in enumerate(evidences))
    def _parse(response: str) -> List[str]:
        response = re.search(r"<ANSWER>(.*?)</ANSWER>", response, re.DOTALL)
        assert response is not None, "没有找到<ANSWER>标签包裹的内容。"
        response = response.group(1).strip()
        groups = []
        seen_ids = set()
        for line in response.strip().splitlines():
            ids = line.strip().split()
            for _id in ids: assert re.match(r"^\d+$", _id), f"输出格式错误：输出了数字id之外的其他字符: {_id}"
            unique_ids = [int(id_) for id_ in ids if id_ not in seen_ids]
            if unique_ids:
                groups.append(unique_ids)
                seen_ids.update(unique_ids)
        unseen_ids = set(range(1, 1 + len(evidences))) - seen_ids
        unique_evidences = []
        for group in groups:
            first_id = int(group[0]) - 1
            unique_evidences.append(list(evidences)[first_id])
        for unseen_id in unseen_ids:
            first_id = int(unseen_id) - 1
            unique_evidences.append(list(evidences)[first_id])
        return unique_evidences
    return await call_chatbot_with_retry(
        llm_instruct, formatter,
        sys_prompt, f"需要处理的论据列表如下：\n---\n{numbered_evidences}\n---",
        hook=_parse, handle_hook_exceptions=(AssertionError, TypeError), max_retries=3,
    )


async def find_best_matches(source_texts: List[str], texts_to_match: List[str]) -> List[Tuple[str, str]]:
    # 为每个列表的项加上唯一ID
    texts_to_match_formatted = "\n".join(f"  R_{i + 1}: {text}" for i, text in enumerate(texts_to_match))
    source_texts_formatted = "\n".join(f"  S_{i + 1}: {text}" for i, text in enumerate(source_texts))

    prompt = f"""你是一个精准的语义匹配引擎。你的任务是从“源列表(S)”中，为“待匹配列表(R)”里的每一项找到与之**完全匹配或语义上非常相似**的匹配项。

## 任务指令
1.  仔细阅读下面提供的两个列表：待匹配列表(R)和源列表(S)。
2.  对于 R 中的**每一项**（如 R_500），请在 S 中找到与之含义最相同、同时约束条件（例如时间）相同的匹配项（如 S_600）。
    -   如果对于某个 R_ID，在 S 列表中找不到任何合适的匹配项，请{'设为 NONE' if len(source_texts) + len(texts_to_match) < 500 else '跳过，不输出该项'}。
    -   如果存在多个合适的匹配项，选最匹配的一个，不要输出多个S_。
3.  你的答案必须包裹在<ANSWER>和</ANSWER>内。其中每一行是待匹配ID和找到的源ID，逗号隔开。

### 输出示例
<ANSWER>
R_1, S_5
R_3, S_4
R_6, S_2
</ANSWER>
"""
    def _parse(response: str) -> Dict[str, str]:
        match_map_by_id = {}
        answer_match = re.search(r"<ANSWER>(.+)</?ANSWER>", response, re.DOTALL)
        assert answer_match is not None, "输出格式错误：未找到<ANSWER>和</ANSWER>标签包裹的内容。"
        answer_content = answer_match.group(1).strip()
        for line in answer_content.splitlines():
            if "NONE" in line or "无" in line:
                continue
            res = re.search(r"(R_\d+).+(S_\d+)", line.strip())
            assert res is not None, f"输出格式错误。错误行: {line}"
            r_id, s_id = res.group(1), res.group(2)
            match_map_by_id[r_id] = s_id
        return match_map_by_id
    match_map_by_id = await call_chatbot_with_retry(
        llm_instruct, formatter,
        prompt, f"**待匹配列表(R):**\n---\n{texts_to_match_formatted}\n---\n\n"
                f"**源列表(S):**\n---\n{source_texts_formatted}\n---\n\n",
        hook=_parse, handle_hook_exceptions=(AssertionError, ), max_retries=3,
    )
    evidence_pairs = []
    for i, r_text in enumerate(texts_to_match):
        r_id = f"R_{i + 1}"
        s_id = match_map_by_id.get(r_id)

        if s_id and s_id != "NONE":
            try:
                s_idx = int(s_id.split('_')[1]) - 1
                if 0 <= s_idx < len(source_texts):
                    s_text = source_texts[s_idx]
                    evidence_pairs.append((s_text, r_text))
            except (ValueError, IndexError) as e:
                print(e)
                continue  # 如果ID格式错误，则跳过

    print(f"    -> 配对完成，成功构建了 {len(evidence_pairs)} 对可供判断的论据。")
    return evidence_pairs


async def find_locations_in_outline(outline_content_str: str, evidences_to_find: List[str]) -> Dict[str, str]:
    """在研报大纲中定位论据的位置。"""
    print(f"  - 正在大纲中定位 {len(evidences_to_find)} 条共通论据的位置...")
    if not evidences_to_find:
        return {}

    numbered_evidences = "\n".join(f"EV_{i + 1}: {e}" for i, e in enumerate(evidences_to_find))

    # 为了防止Prompt过长，对outline_content进行简化，只保留关键信息
    outline_json = json.loads(outline_content_str)
    simplified_outline = []

    def simplify_section(section, prefix):
        current_loc = f"{prefix}s{section.get('section_id')}"
        simplified_outline.append(f"章节点: {current_loc}, 标题: \"{section.get('title')}\"")
        if section.get('segments'):
            for i, seg in enumerate(section['segments']):
                seg_loc = f"{current_loc}_p{seg.get('segment_id', i + 1)}"
                if seg.get('evidences'):
                    simplified_outline.append(
                        f"  - 段落点: {seg_loc}, 论据: {json.dumps(seg.get('evidences'), ensure_ascii=False)}")
        if section.get('subsections'):
            for sub in section['subsections']:
                simplify_section(sub, f"{current_loc}_")

    simplify_section(outline_json, "")
    simplified_outline_text = "\n".join(simplified_outline)

    prompt = """你是一名精准的文本定位专家。你的任务是在给定的研报大纲结构中，为一系列指定的论据找到它们的确切位置路径。

## 背景与定义
1.  **研报大纲**: 以下是一份简化版的研报结构，每一行代表一个章节或段落。
2.  **位置路径格式**: 路径由 `s{id}` (章节) 和 `p{id}` (段落) 组成，层层相连。
    - `s{id}`: 代表 `section_id`。
    - `p{id}`: 代表 `segment_id`。
    - **示例**: `s1_s3_p2` 表示在 `section_id=1` 的章节下的 `section_id=3` 的子章节中的 `segment_id=2` 的段落。

## 任务指令
1.  仔细阅读下面提供的 "研报大纲结构"。
2.  对于 "待查找论据列表" 中的每一条论据（以 EV_ID 开头），在大纲的 `evidences` 数组中找到与之**完全匹配或语义上非常相似**的条目。
3.  记录下该条目所属的 "段落点" (例如 `s1_s3_p2`)。
4.  你的答案必须包裹在<ANSWER>和</ANSWER>内。其中每一行是论据的ID和找到的位置路径字符串，逗号隔开。每一条论据ID都必须有对应的路径，不能有遗漏。

### 输出示例
<ANSWER>
EV_1,s1_s2_p1
EV_2,s3_p1
EV_3,s2_s3_p2
</ANSWER>
"""

    def _parse(response: str) -> Dict[str, str]:
        locations_map = {}
        answer_match = re.search(r"<ANSWER>(.+)</?ANSWER>", response, re.DOTALL)
        assert answer_match is not None, "输出格式错误：未找到<ANSWER>和</ANSWER>标签包裹的内容。"
        answer_content = answer_match.group(1).strip()
        for line in answer_content.splitlines():
            parts = line.strip().split(",", 1)
            if len(parts) < 2:
                print(f"输出格式错误：每行应包含论据ID和位置路径，用逗号分隔。错误行: {line}")
            ev_id, location = parts
            assert ev_id.strip().startswith("EV_"), f"输出格式错误：论据ID没有用EV_开头。错误行: {line}"
            locations_map[ev_id.strip()] = location.strip()
        return locations_map

    locations_map_by_id = await call_chatbot_with_retry(
        llm_instruct, formatter,
        prompt, f"**研报大纲结构:**\n---\n{simplified_outline_text}\n---\n"
                f"**待查找论据列表:**\n---\n{numbered_evidences}\n---\n",
        hook=_parse, handle_hook_exceptions=(AssertionError,), max_retries=5,
    )
    locations_map_by_text = {}
    for i, evidence in enumerate(evidences_to_find):
        ev_id = f"EV_{i + 1}"
        location = locations_map_by_id[ev_id]
        locations_map_by_text[evidence] = location
    print("    -> 定位完成。")
    return locations_map_by_text

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

def save_result_with_lock(result_data: Dict, output_json_path: Path) -> bool:
    """
    使用文件锁机制安全地保存单个结果到JSON文件。

    Args:
        result_data: 要保存的结果数据字典
        output_json_path: 输出JSON文件的路径

    Returns:
        bool: 保存成功返回 True，失败返回 False
    """
    if result_data is None:
        return False

    lock_path = output_json_path.with_suffix('.lock')

    # 创建文件锁，超时时间为 60 秒
    with FileLock(str(lock_path), timeout=60) as lock:
        try:
            # 读取现有的 JSON 数据
            results_cache = {}
            if output_json_path.exists():
                # try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                # except (json.JSONDecodeError, KeyError) as e:
                #     print(f"  ⚠️  警告: 读取现有结果文件时出错: {e}，将创建新文件。")
                results_cache = {(res['stock_code'], res['old_report'], res['new_report']): res for res in existing_results}

            # 更新或插入新结果
            results_cache[(result_data['stock_code'], result_data['old_report'], result_data['new_report'])] = result_data

            # 将更新后的结果写回文件
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(list(results_cache.values()), f, ensure_ascii=False, indent=4)

            print(f"  ✓ 结果已安全保存到 {output_json_path.name}")
            return True
        except Exception as e:
            print(f"  ✗ 保存结果时出错: {e}")
            traceback.print_exc()
        finally:
            lock.release(force=True)
    return False


async def process_single_stock_pair(stock_code: str, old_path: Path, new_path: Path,
                                    long_term_dir: Path, output_json_path: Path,
                                    results_cache: Dict, semaphore: asyncio.Semaphore) -> Optional[Dict]:
    """
    处理单个股票对的函数，可并发执行。

    Args:
        stock_code: 股票代码
        old_path: 旧报告路径
        new_path: 新报告路径
        long_term_dir: 长期记忆目录
        output_json_path: 输出JSON文件路径
        results_cache: 结果缓存字典
        semaphore: 用于限制并发数的信号量

    Returns:
        该股票的处理结果数据字典
    """

    async with semaphore:
        try:
            print(f"\n======= 正在处理股票: {stock_code} =======", flush=True)

            # ===== 并发提取两份报告的论据 =====
            evidences_old, evidences_new = await asyncio.gather(
                extract_unique_evidences_from_pdf(old_path, long_term_dir / "demonstration"),
                extract_unique_evidences_from_pdf(new_path, long_term_dir / "demonstration")
            )
            common_evidences_texts = await find_best_matches(evidences_old, evidences_new)

            common_evidences_with_locs = []
            assert len(common_evidences_texts) > 0, "!!! 未找到任何共通论据"
            if common_evidences_texts:
                print(f"\n  --- 开始为 {len(common_evidences_texts)} 条共通论据定位 ---", flush=True)
                outline_old_path = long_term_dir / "demonstration" / f"{old_path.stem}_outline.json"
                outline_new_path = long_term_dir / "demonstration" / f"{new_path.stem}_outline.json"

                outline_old_content = outline_old_path.read_text(encoding='utf-8')
                outline_new_content = outline_new_path.read_text(encoding='utf-8')

                # ===== 并发定位两份报告中论据的位置 =====
                locations_old, locations_new = await asyncio.gather(
                    find_locations_in_outline(outline_old_content, [e[0] for e in common_evidences_texts]),
                    find_locations_in_outline(outline_new_content, [e[1] for e in common_evidences_texts])
                )
                for old_text, new_text in common_evidences_texts:
                    common_evidences_with_locs.append({
                        "text": (old_text, new_text),
                        "location_old": locations_old.get(old_text, "NONE"),
                        "location_new": locations_new.get(new_text, "NONE")
                    })

            # 过滤掉任何包含 "NONE" 的共通论据
            initial_count = len(common_evidences_with_locs)

            filtered_common_evidences = [
                item for item in common_evidences_with_locs
                if "NONE" not in item.get("location_old", "NONE") and \
                   "NONE" not in item.get("location_new", "NONE")
            ]

            filtered_count = len(filtered_common_evidences)
            if initial_count > filtered_count:
                print(f"  - 过滤完成：移除了 {initial_count - filtered_count} 条无法在两份报告中同时定位的共通论据。", flush=True)

            result_data = {
                "stock_code": stock_code, "old_report": old_path.name, "new_report": new_path.name,
                "old_evidence_count": len(evidences_old), "new_evidence_count": len(evidences_new),
                "common_evidence_count": filtered_count,
                # "common_evidences": filtered_common_evidences
            }

            print(f"--- 股票 {stock_code} 处理完成 ---", flush=True)

            # 立即使用文件锁保存结果
            save_result_with_lock(result_data, output_json_path)

            return result_data

        except Exception as e:
            traceback.print_exc()
            print(f"!!! 处理股票 {stock_code} 时发生错误: {e}，跳过该股票。", flush=True)
            return None

def _process_single_stock_pair(stock_code: str, old_path: Path, new_path: Path,
                                    long_term_dir: Path, output_json_path: Path,
                                    results_cache: Dict) -> Optional[Dict]:
    return asyncio.run(process_single_stock_pair(stock_code, old_path, new_path, long_term_dir, output_json_path, results_cache))

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
    results_cache = {(res['stock_code'], res['old_report'], res['new_report']): res for res in existing_results}
    print(f"检测到 {len(results_cache)} 个已保存的结果。")

    print(f"\n--- 步骤 2: 开始批量处理研报对（并发模式，最大并发数: {MAX_CONCURRENT}）... ---")

    def _completed(k):
        cached_data = results_cache.get(k)
        if cached_data and "common_evidences" in cached_data:
            first_common = cached_data["common_evidences"][0] if cached_data["common_evidences"] else {}
            if isinstance(first_common, dict) and "location_old" in first_common:
                print(f"--- 股票 {k} 已被完整处理（包括定位），完全跳过 ---", flush=True)
                return True
        return False

    # 创建信号量，用于限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # results = []
    # for stock_code, old_path, new_path in report_pairs:
    #     if not _completed((stock_code, old_path.name, new_path.name)):
    #         res = await process_single_stock_pair(stock_code, old_path, new_path, long_term_dir, output_json_path, results_cache, semaphore)
    #         results.append(res)

    # 创建并发任务列表
    tasks = [
        process_single_stock_pair(stock_code, old_path, new_path, long_term_dir, output_json_path, results_cache, semaphore)
        for stock_code, old_path, new_path in report_pairs if not _completed((stock_code, old_path.name, new_path.name))
    ]

    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # # 使用线程池并发执行所有任务
    # results = []
    # with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
    #     # 提交所有任务
    #     futures = {
    #         executor.submit(
    #             _process_single_stock_pair,
    #             stock_code, old_path, new_path,
    #             long_term_dir, output_json_path, results_cache
    #         ): stock_code
    #         for stock_code, old_path, new_path in report_pairs if not _completed((stock_code, old_path.name, new_path.name))
    #     }
    #     # 收集结果
    #     for future in as_completed(futures):
    #         stock_code = futures[future]
    #         try:
    #             result = future.result()
    #             results.append(result)
    #         except Exception as e:
    #             print(f"任务执行中发生异常 (股票 {stock_code}): {e}")
    #             traceback.print_exc()
    #             results.append(None)

    # 处理结果统计
    successful_count = 0
    failed_count = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"任务执行中发生异常: {result}")
            failed_count += 1
            continue
        if result is not None and isinstance(result, dict):
            successful_count += 1
            results_cache[result['stock_code']] = result
        else:
            failed_count += 1

    print(f"\n✓ 成功处理: {successful_count} 个任务")
    print(f"✗ 失败处理: {failed_count} 个任务")
    print(f"(所有结果已在处理过程中实时保存)")

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