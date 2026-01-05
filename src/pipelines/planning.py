from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from pathlib import Path

from agentscope.model import ChatModelBase

from memory.working import Section
from prompt import prompt_dict
from utils.file_converter import pdf_to_markdown, markdown_to_sections
from utils.instance import create_agent_formatter


async def planning(task_desc: str, demo_pdf_path: Path, short_term_dir: Path,
                   llm_reasoning: ChatModelBase, llm_instruct: ChatModelBase=None, formatter=None):
    if formatter is None:
        formatter = create_agent_formatter()
    if llm_instruct is None:
        llm_instruct = llm_reasoning
    demo_date, demo_name = demo_pdf_path.name.split(".")[0].split("_")[-2:]
    demo_md_path = short_term_dir / f"demonstration" / (demo_pdf_path.name.split(".")[0] + ".md")
    if not demo_md_path.exists():
        final_text, images = pdf_to_markdown(demo_pdf_path, demo_md_path)
    manuscript: Section = markdown_to_sections(demo_md_path)

    async def dfs_outline(section: Section, parent_id=None):
        if section.subsections is None:
            return
        for subsection in section.subsections:
            section_id = ((parent_id + ".") if parent_id else "") + str(subsection.section_id)
            print(f"\n====== 开始总结章节 {section_id} ======\n")
            await dfs_outline(subsection)
            if subsection.segments:
                decomposer_input = await formatter.format([
                    Msg("system", prompt_dict["decompose"],"system"),
                    Msg("user", subsection.segments[0].reference.replace("<SEP>", ""), "user", )
                ])
                for i in range(10):
                    try:
                        decomposed_content = await llm_instruct(decomposer_input)
                        break
                    except Exception as e:
                        print(e)
                segments = Msg("assistant", decomposed_content.content, "assistant").get_text_content().split("<SEP>")
                subsection.segments = []
                for i, segment in enumerate(segments):
                    planner_input = [
                        Msg("system", prompt_dict["plan_outline"],"system"),
                        Msg(
                            name="user",
                            content=f"当前任务：{task_desc}\n\n为实现当前任务，我找到了某机构在{demo_date}撰写的一份研报，名为{demo_name}。"
                                    f"下文将附上从中摘出的一段参考片段，请你考虑时间差和公司异同，撰写一份用于当前新任务的撰写模版和要求。\n\n"
                                    f"参考片段如下：\n\n{segment}",
                            role="user",
                        )
                    ]
                    # outline_msg = await planner(planner_input)
                    print(segment, flush=True)
                    for i in range(10):
                        try:
                            _input = await formatter.format(planner_input)
                            outline_msg = await llm_reasoning(_input)
                            # print(outline_msg.get_text_content())
                            outline_msg = Msg("assistant", outline_msg.content, "assistant")
                            subsection.segments.append(subsection.parse(outline_msg.get_text_content()))
                            subsection.segments[-1].reference = segment
                            break
                        except AssertionError as e:
                            print(e)
                            planner_input += [
                                outline_msg,
                                Msg("user", str(e), "user")
                            ]
            print(subsection.read(True, True, True, True, False, False))

    outline_json_pth = short_term_dir / "outline.json"
    if not outline_json_pth.exists():
        await dfs_outline(manuscript)
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True, fold_other=False)
        print(outline)
        outline_json_pth.write_text(manuscript.to_json(ensure_ascii=False))
    else:
        # outline = outline_md_pth.read_text()
        manuscript = Section.from_json(outline_json_pth.read_text())
        outline = manuscript.read(read_subsections=True, with_reference=True, with_content=True, with_evidence=True, fold_other=False)
        print(outline)
