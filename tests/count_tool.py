import re
from pathlib import Path
from collections import defaultdict

log_dir = Path(r"E:\code\vscode\financial-report-agent\tests")


agent_pattern = re.compile(r'^(Searcher|Writer):\s*(.*)$')
stop_pattern = re.compile(r'^(Searcher|Writer|system):')
type_pattern = re.compile(r'"type"\s*:\s*"([^"]+)"')
name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')


def extract_json_from_pos(lines, start_idx, first_text=""):
    """
    从 start_idx 开始提取一个 JSON 块。
    first_text 是当前行里 agent: 后面的剩余内容，可能本身就以 { 开头。
    返回 (json_text, next_idx)；若没提取到，返回 (None, 原位置后继)
    """
    # 情况1：当前行后半部分就已经是 JSON 开头
    if first_text.strip().startswith("{"):
        buf = [first_text[first_text.index("{"):]]
        brace = buf[0].count("{") - buf[0].count("}")
        i = start_idx + 1

        while i < len(lines) and brace > 0:
            buf.append(lines[i])
            brace += lines[i].count("{") - lines[i].count("}")
            i += 1

        if brace == 0:
            return "\n".join(buf), i
        return None, start_idx + 1

    # 情况2：当前行只是说明文字，往下找紧随的 JSON
    i = start_idx + 1

    while i < len(lines):
        line = lines[i]

        # 遇到新的角色段/系统段，停止，避免串到下一段
        if stop_pattern.match(line):
            return None, i

        # 跳过空行
        if not line.strip():
            i += 1
            continue

        # 找到 JSON 开头
        if line.lstrip().startswith("{"):
            buf = [line]
            brace = line.count("{") - line.count("}")
            i += 1

            while i < len(lines) and brace > 0:
                buf.append(lines[i])
                brace += lines[i].count("{") - lines[i].count("}")
                i += 1

            if brace == 0:
                return "\n".join(buf), i
            return None, i

        # 非空且不是 JSON，继续往下找，兼容“agent 先说一段再贴 JSON”
        i += 1

    return None, i


def count_tools_in_text(log_text: str):
    counts = defaultdict(lambda: defaultdict(int))
    lines = log_text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = agent_pattern.match(line)
        if not m:
            i += 1
            continue

        agent, rest = m.groups()
        json_text, next_i = extract_json_from_pos(lines, i, rest)

        if json_text:
            type_match = type_pattern.search(json_text)
            name_match = name_pattern.search(json_text)

            if type_match and type_match.group(1) == "tool_use" and name_match:
                tool_name = name_match.group(1)
                counts[agent][tool_name] += 1

        # 关键：无论是否找到 JSON，都推进，避免死循环
        i = max(next_i, i + 1)

    return counts


def merge_counts(total_counts, file_counts):
    for agent, tools in file_counts.items():
        for tool, cnt in tools.items():
            total_counts[agent][tool] += cnt


def print_counts(title, counts):
    print(f"\n===== {title} =====")
    for agent in ["Searcher", "Writer"]:
        print(f"{agent}:")
        if not counts[agent]:
            print("  无工具调用")
        else:
            for tool, cnt in sorted(counts[agent].items()):
                print(f"  {tool}: {cnt}")


def main():
    if not log_dir.exists():
        print(f"目录不存在: {log_dir}")
        return

    txt_files = sorted(log_dir.rglob("*.txt"))
    if not txt_files:
        print(f"目录下没有 txt 文件: {log_dir}")
        return

    total_counts = defaultdict(lambda: defaultdict(int))

    for txt_file in txt_files:
        try:
            log_text = txt_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log_text = txt_file.read_text(encoding="gbk", errors="ignore")

        file_counts = count_tools_in_text(log_text)
        print_counts(txt_file.name, file_counts)
        merge_counts(total_counts, file_counts)

    print_counts("总统计", total_counts)


if __name__ == "__main__":
    main()