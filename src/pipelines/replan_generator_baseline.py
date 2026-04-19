# 文件名: src/utils/replan_generator.py 

from __future__ import annotations
import re
from typing import List, Dict, Any

from src.memory.feedback_models import Feedback
from src.memory.working import Section
from src.utils.outline_navigator import find_evidence_by_id, find_segment_by_id, find_parent_section_and_index
from src.utils.call_with_retry import call_chatbot_with_retry
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
import json

# 针对“单点 Evidence 错误”的重规划函数
async def replan_for_single_evidence_error(
    feedback: Feedback, 
    manuscript: Section, 
    llm_planner: ChatModelBase, 
    formatter: FormatterBase) -> Dict[str, Any]:
    """
    针对单点的 Evidence 搜索失败，生成修正指令。
    """
    print(f"  - Re-planning for Single Evidence Error: {feedback.evidence_id}")

    # 步骤 1: 通过 evidence_id 查找 Evidence 对象
    target_evidence = find_evidence_by_id(manuscript, feedback.evidence_id)
    if not target_evidence:
        print(f"    ! 警告: 无法找到 Evidence '{feedback.evidence_id}'，跳过此反馈。")
        return None

    # 从对象中获取 text
    evidence_text = target_evidence.text

    # 步骤 2: 构建 Re-planning Prompt (Baseline版)
    prompt = f"""
    你是一个金融分析专家。在生成研报的过程中，一个论据（evidence）搜索失败了。

    **失败的论据描述:**
    "{evidence_text}"

    **任务指令:**
    请你根据这个失败的论据描述，提出一个修正后的、更可能被成功搜索到的新论据。

    **输出格式:**
    请直接返回修正后的论据文本字符串，不要包含任何其他解释或标签。
    """
    
    try:
        # 步骤 3: 调用 LLM 生成修正后的 evidence 文本
        response_msg = await call_chatbot_with_retry(
                llm_planner, 
                formatter,
                sys_prompt=prompt,      # 传入你的完整 Prompt
                user_prompt="",         # 必须传入一个空的 user_prompt 字符串
                hook=None               
            )
        # [修复] 直接使用 response_msg 作为字符串（因为 call_chatbot_with_retry 返回字符串）
        corrected_evidence_text = response_msg.strip()
        
        if not corrected_evidence_text:
            print(f"    ! LLM 未能为 '{evidence_text}' 生成有效的修正。")
            return None
            
        print(f"    -> LLM 建议将 '{evidence_text[:30]}...' 修正为 '{corrected_evidence_text[:30]}...'")

        # [新逻辑] 步骤 4: 构建针对单个 Evidence 的修正指令
        instruction = {
            "type": "modify_evidence_field",
            "evidence_id": feedback.evidence_id,
            "field": "text", # 我们要修改的是 text 字段
            "value": corrected_evidence_text,
            "reason": f"对失败的论据 '{evidence_text[:20]}...' 进行了 Baseline 修正。"
        }
        
        return instruction

    except Exception as e:
        print(f"    ! 在为单点错误重规划时发生异常: {e}")
        return None
    
async def replan_for_writer_error(
    feedback: Feedback, 
    manuscript: Section, 
    llm_planner: ChatModelBase, 
    formatter: FormatterBase) -> List[Dict[str, Any]]:
    """
    针对 Writer 撰写/填充失败，生成修正指令。
    这是 Baseline 版本，让 LLM 自主决定修正 template 还是 requirements。
    """
    segment_id = feedback.segment_id
    if not segment_id:
        return []

    print(f"  - Re-planning for Writer Error in Segment: {segment_id}")

    # 1. 定位到出错的 Segment
    target_segment = find_segment_by_id(manuscript, segment_id)
    if not target_segment:
        print(f"    ! 警告: 无法找到 Segment '{segment_id}'，跳过此反馈。")
        return []

    # 2. 构建 Re-planning Prompt (最简版)
    prompt = f"""
    你是一个金融分析专家。在生成研报的过程中，写作 Agent 在处理一个段落时遇到了问题。

    **段落主题:**
    {target_segment.topic}

    **原始写作模板 (Template):**
    ```
    {target_segment.template}
    ```

    **原始写作要求 (Requirements):**
    ```
    {target_segment.requirements}
    ```

    **遇到的问题描述:**
    {feedback.message}

    **任务指令:**
    根据上述问题，请你同时修正**写作模板 (template)**和**写作要求 (requirements)**，使它们更合理、更易于执行。

    **输出格式:**
    请严格按照以下格式返回修正后的内容，不要包含任何其他解释或标签：
    <template>
    [这里是修正后的写作模板]
    </template>
    <requirement>
    [这里是修正后的写作要求]
    </requirement>
    """
    
    try:
        # 3. 调用 LLM 生成修正后的 template 和 requirements
        response_msg = await call_chatbot_with_retry(
            llm_planner, 
            formatter,
            sys_prompt=prompt,
            user_prompt="",
        )
        corrected_content = response_msg.strip()
        
        # 4. 解析 LLM 返回的内容
        template_match = re.search(r"<template>(.*?)</template>", corrected_content, re.DOTALL)
        requirement_match = re.search(r"<requirement>(.*?)</requirement>", corrected_content, re.DOTALL)

        if not template_match or not requirement_match:
            print(f"    ! LLM 未能按格式返回修正内容。")
            return []

        new_template = template_match.group(1).strip()
        new_requirements = requirement_match.group(1).strip()
        
        print(f"    -> LLM 建议修正 Template 和 Requirements。")

        # 5. 构建两条修正指令
        instructions = []
        # 指令1：修改 template
        instructions.append({
            "type": "modify_segment_field",
            "segment_id": segment_id,
            "field": "template",
            "value": new_template,
            "reason": f"根据 Writer 反馈 '{feedback.message[:20]}...' 进行 Baseline 修正。"
        })
        # 指令2：修改 requirements
        instructions.append({
            "type": "modify_segment_field",
            "segment_id": segment_id,
            "field": "requirements",
            "value": new_requirements,
            "reason": f"根据 Writer 反馈 '{feedback.message[:20]}...' 进行 Baseline 修正。"
        })
        
        return instructions

    except Exception as e:
        print(f"    ! 在为 Writer 错误重规划时发生异常: {e}")
        return []

async def replan_for_regional_segment_error(
    feedback: Dict[str, Any], # 这是一个包含 segment_id, reason, related_feedbacks 的字典
    manuscript: Section, 
    llm_planner: ChatModelBase, 
    formatter: FormatterBase) -> List[Dict[str, Any]]:
    """
    针对区域性的 Segment 失败，生成大规模修正指令。
    这是 Baseline 版本，让 LLM 根据失败信息自主决定如何整体修正。
    """
    segment_id = feedback.get("segment_id")
    if not segment_id:
        return []

    print(f"  - Re-planning for Regional Segment Error: {segment_id}")

    # 1. 定位到出错的 Segment
    target_segment = find_segment_by_id(manuscript, segment_id)
    if not target_segment:
        print(f"    ! 警告: 无法找到 Segment '{segment_id}'，跳过此反馈。")
        return []

    # 2. 构建 Re-planning Prompt (Baseline版)
    # 将相关的原始 feedback 信息格式化，供 LLM 参考
    related_feedback_summary = "\n".join(
        [f"- 论据 '{fb.get('evidence_text', '未知')}' 失败，原因: {fb.get('message', '未知')}" for fb in feedback.get("related_feedbacks", [])]
    )

    prompt = f"""
    你是一个金融分析专家，负责优化研报的规划大纲。现在，一个段落（Segment）在执行时遇到了系统性问题。

    **段落主题:**
    {target_segment.topic}

    **诊断信息:**
    {feedback.get('reason', '该段落的大部分论据搜索失败。')}

    **失败的论据详情:**
    {related_feedback_summary}

    **原始规划:**
    - **原始写作模板 (Template):**
    ```
    {target_segment.template}
    ```
    - **原始论据列表 (Evidences):**
    ```
    {[ev.text for ev in target_segment.evidences]}
    ```

    **任务指令:**
    根据上述诊断信息，请你**重新思考**这个段落的规划。这很可能是一个规划层面的错误。
    请你**大规模修正**或**完全重新生成**该段落的**写作模板 (template)**和**所有论据 (evidences)**，使其更合理、更具可执行性。

    **输出格式:**
    请严格按照以下格式返回修正后的内容，不要包含任何其他解释或标签：
    <template>
    [这里是全新的、修正后的写作模板]
    </template>
    <evidence>
    [这里是全新的、修正后的论据列表，每条论据用分号;隔开，如果它是静态论据在论据后标记 (static) ]
    </evidence>
    """
    
    try:
        # 3. 调用 LLM 生成修正后的 template 和 evidences
        response_msg = await call_chatbot_with_retry(
            llm_planner, 
            formatter,
            sys_prompt=prompt,
            user_prompt="",
        )
        corrected_content = response_msg.strip()
        print(f"LLM 返回的原始内容:\n{corrected_content}\n")
        
        # 4. 解析 LLM 返回的内容
        template_match = re.search(r"<template>(.*?)</template>", corrected_content, re.DOTALL)
        evidence_match = re.search(r"<evidence>(.*?)</evidence>", corrected_content, re.DOTALL)

        if not template_match or not evidence_match:
            print(f"    ! LLM 未能按格式返回修正内容。")
            return []

        new_template = template_match.group(1).strip()
        new_evidences_text = evidence_match.group(1).strip()
        
        # 将新的 evidence 文本解析为 Evidence 对象列表（与 working.py 中的 parse 逻辑类似）
        new_evidence_list_of_dicts = []
        raw_evidences = new_evidences_text.replace("\n", "").replace(";", "；").split("；")
        for e_str in raw_evidences:
            e_str = e_str.strip()
            if not e_str: continue
            is_static, static_value = False, None
            # ... (这里可以复用 working.py 中解析 (static)[value] 的逻辑)
            if e_str.lower().endswith("(static)") or e_str.endswith("(静态)"):
                is_static = True
                e_str = re.sub(r'\s*\((static|静态)\)\s*$', '', e_str, flags=re.IGNORECASE).strip()
            new_evidence_list_of_dicts.append({"text": e_str, "is_static": is_static, "value": static_value})

        print(f"    -> LLM 建议对 Segment '{segment_id}' 进行大规模修正。")

        # 5. 构建两条修正指令
        instructions = [
            {
                "type": "modify_segment_field",
                "segment_id": segment_id,
                "field": "template",
                "value": new_template,
                "reason": f"对区域性失败的 Segment 进行 Baseline 整体修正。"
            },
            {
                "type": "modify_segment_field",
                "segment_id": segment_id,
                "field": "evidences",
                "value": new_evidence_list_of_dicts, # value 是一个字典列表
                "reason": f"对区域性失败的 Segment 进行 Baseline 整体修正。"
            }
        ]

        print(f"    -> 为 Segment '{segment_id}' 生成了 {len(instructions)} 条修正指令。")
        return instructions

    except Exception as e:
        print(f"    ! 在为区域性错误重规划时发生异常: {e}")
        return []

async def replan_for_macro_incompleteness(
    feedback: Feedback, 
    manuscript: Section, 
    llm_planner: ChatModelBase, 
    formatter: FormatterBase) -> List[Dict[str, Any]]:
    """
    针对宏观完整性缺失，生成新增 Segment 的指令。
    """
    # 1. 定位插入锚点 (这是你要求的关键点)
    parent_segment_id = feedback.segment_id
    target_segment = find_segment_by_id(manuscript, parent_segment_id)
    
    # 构建锚点上下文 (给LLM看的)
    anchor_context = "无特定上下文"
    if target_segment:
        anchor_context = f"插入点位置主题: '{target_segment.topic}'\n参考内容: {target_segment.reference}"
    else:
        print(f"  ! 警告: 无法定位到插入锚点 '{parent_segment_id}'，将作为章节末尾追加。")

    # 2. 构建 Prompt
    prompt = f"""
    你是一位资深的金融研报总架构师。当前研报大纲在逻辑完整性上存在缺失。

    **插入点上下文 (请确保新生成的段落在此之后衔接自然):**
    {anchor_context}

    **缺失问题描述:**
    {feedback.message}

    **任务指令:**
    
    根据“缺失问题描述”，请你**创造并生成一个或多个全新的段落（Segment）**使其在逻辑上能够自然地衔接在上述“插入点位置”之后。。
    对于每一个你认为需要新增的 Segment，请完整地规划出它的 **topic, template, evidences, 和 requirements**。

    **输出格式:**
    请严格按照以下格式返回你生成的新 Segment(s)。如果需要新增多个，请将它们并列放在 <new_segments> 标签内。
    <new_segments>
      <segment>
        <topic>[新Segment的主题]</topic>
        <template>[新Segment的写作模板]</template>
        <evidence>[新Segment的论据列表，用分号;隔开，如果它是静态论据在论据后标记 (static)]</evidence>
        <requirement>[新Segment的写作要求]</requirement>
      </segment>
    </new_segments>
    """
    
    try:
        # 3. 调用 LLM 生成新 Segment(s) 的内容
        response_msg = await call_chatbot_with_retry(
            llm_planner, 
            formatter,
            sys_prompt=prompt,
            user_prompt="",
        )
        llm_response = response_msg.strip()
        
        # 4. 解析 LLM 返回的新 Segment 内容
        new_segments_matches = re.findall(r"<segment>(.*?)</segment>", llm_response, re.DOTALL)
        if not new_segments_matches:
            print(f"    ! LLM 未能生成任何新的 Segment。")
            return []

        instructions = []
        for segment_content in new_segments_matches:
            topic_match = re.search(r"<topic>(.*?)</topic>", segment_content, re.DOTALL)
            template_match = re.search(r"<template>(.*?)</template>", segment_content, re.DOTALL)
            evidence_match = re.search(r"<evidence>(.*?)</evidence>", segment_content, re.DOTALL)
            requirement_match = re.search(r"<requirement>(.*?)</requirement>", segment_content, re.DOTALL)
            
            if not all([topic_match, template_match, evidence_match, requirement_match]):
                print("    ! LLM 生成的新 Segment 格式不完整，跳过。")
                continue

            # 将 evidence 文本解析为字典列表，以便 Executor 处理
            new_evidences_text = evidence_match.group(1).strip()
            new_evidence_list_of_dicts = []
            raw_evidences = new_evidences_text.replace("\n", "").replace(";", "；").split("；")
            for e_str in raw_evidences:
                e_str = e_str.strip()
                if not e_str: continue
                is_static, static_value = False, None
                # ... (复用 working.py 中解析 (static)[value] 的逻辑) ...
                if e_str.lower().endswith("(static)") or e_str.endswith("(静态)"):
                    is_static = True
                    e_str = re.sub(r'\s*\((static|静态)\)\s*$', '', e_str, flags=re.IGNORECASE).strip()
                new_evidence_list_of_dicts.append({"text": e_str, "is_static": is_static, "value": static_value})

            # 5. 构建 add_segment 指令
            instruction = {
                "type": "add_segment",
                "parent_segment_id": parent_segment_id, # [新逻辑] 直接使用 feedback 中的 segment_id
                "new_segment_data": {
                    "topic": topic_match.group(1).strip(),
                    "template": template_match.group(1).strip(),
                    "requirements": requirement_match.group(1).strip(),
                    "evidences": new_evidence_list_of_dicts
                },
                "reason": f"弥补宏观完整性缺失: {feedback.message[:20]}..."
            }
            instructions.append(instruction)

        print(f"    -> LLM 建议在 '{parent_segment_id}' 之后新增 {len(instructions)} 个 Segment 来弥补完整性。")
        return instructions

    except Exception as e:
        print(f"    ! 在为宏观完整性缺失重规划时发生异常: {e}")
        return []
    

async def replan_for_cross_segment_contradiction(
    feedback: Feedback, 
    manuscript: Section, 
    llm_planner: ChatModelBase, 
    formatter: FormatterBase) -> List[Dict[str, Any]]:
    """
    针对跨 Segment 的逻辑矛盾，生成修正指令。
    """
    segment_id_1 = feedback.segment_id
    segment_id_2 = feedback.segment_id_2

    if not all([segment_id_1, segment_id_2]):
        print("  ! 警告: '跨 Segment 矛盾' 的反馈中缺少必要的 segment_id，无法处理。")
        return []

    print(f"  - Re-planning for Cross-Segment Contradiction: {segment_id_1} vs {segment_id_2}")

    # 1. 定位到两个矛盾的 Segment
    segment_1 = find_segment_by_id(manuscript, segment_id_1)
    segment_2 = find_segment_by_id(manuscript, segment_id_2)

    if not all([segment_1, segment_2]):
        print(f"    ! 警告: 无法在 outline 中找到一个或多个矛盾的 Segment，跳过此反馈。")
        return []

    # 2. 构建 Re-planning Prompt
    prompt = f"""
    你是一位顶级的金融研报总编辑，负责确保报告的逻辑一致性。在审查一份研报大纲时，发现以下两个段落（Segment）之间存在事实或逻辑上的矛盾。

    **矛盾问题描述:**
    {feedback.message}

    ---
    **段落 A (ID: {segment_id_1}):**
    - **主题:** {segment_1.topic}
    - **写作模板:** {segment_1.template}
    - **论据列表:** {[ev.text for ev in segment_1.evidences] if segment_1.evidences else "无"}

    **段落 B (ID: {segment_id_2}):**
    - **主题:** {segment_2.topic}
    - **写作模板:** {segment_2.template}
    - **论据列表:** {[ev.text for ev in segment_2.evidences] if segment_2.evidences else "无"}
    ---

    **任务指令:**
    请你分析这两个段落的矛盾之处，并提出修正方案。你可以选择修改其中一个或两个段落的 **template** 或 **evidences**，使其逻辑保持一致。

    **输出格式:**
    请严格按照以下格式返回你认为需要修正的段落信息。对于不需要修改的段落，请不要包含在输出中。
    <corrections>
      <segment_A>
        <template>[这里是修正后的段落A的写作模板]</template>
        <evidence>[这里是修正后的段落A的论据列表，用分号;隔开，如果它是静态论据在论据后标记 (static)]</evidence>
      </segment_A>
      <segment_B>
        <template>[这里是修正后的段落B的写作模板]</template>
        <evidence>[这里是修正后的段落B的论据列表，用分号;隔开，如果它是静态论据在论据后标记 (static)]</evidence>
      </segment_B>
    </corrections>
    """
    
    try:
        # 3. 调用 LLM 生成修正内容
        response_msg = await call_chatbot_with_retry(
            llm_planner, formatter, sys_prompt=prompt, user_prompt=""
        )
        llm_response = response_msg.strip()
        
        instructions = []
        
        # 4. 解析 LLM 返回的修正内容
        # 检查是否需要修正 Segment A
        segment_a_match = re.search(r"<segment_A>(.*?)</segment_A>", llm_response, re.DOTALL)
        if segment_a_match:
            content = segment_a_match.group(1)
            template_match = re.search(r"<template>(.*?)</template>", content, re.DOTALL)
            evidence_match = re.search(r"<evidence>(.*?)</evidence>", content, re.DOTALL)
            if template_match:
                instructions.append({
                    "type": "modify_segment_field", "segment_id": segment_id_1,
                    "field": "template", "value": template_match.group(1).strip()
                })
            if evidence_match:
                evidence_text = evidence_match.group(1).strip()
                evidence_list_of_dicts = []
                raw_evidences = evidence_text.replace("\n", "").replace(";", "；").split("；")
                for e_str in raw_evidences:
                    e_str = e_str.strip()
                    if not e_str:
                        continue
                    is_static, static_value = False, None
                    if e_str.lower().endswith("(static)") or e_str.endswith("(静态)"):
                        is_static = True
                        e_str = re.sub(r'\s*\((static|静态)\)\s*$', '', e_str, flags=re.IGNORECASE).strip()
                    evidence_list_of_dicts.append({"text": e_str, "is_static": is_static, "value": static_value})

                instructions.append({
                    "type": "modify_segment_field",
                    "segment_id": segment_id_1,
                    "field": "evidences",
                    "value": evidence_list_of_dicts
                })
            print(f"    -> LLM 建议修正 Segment '{segment_id_1}'。")

        # 检查是否需要修正 Segment B
        segment_b_match = re.search(r"<segment_B>(.*?)</segment_B>", llm_response, re.DOTALL)
        if segment_b_match:
            content = segment_b_match.group(1)
            template_match = re.search(r"<template>(.*?)</template>", content, re.DOTALL)
            evidence_match = re.search(r"<evidence>(.*?)</evidence>", content, re.DOTALL)
            if template_match:
                instructions.append({
                    "type": "modify_segment_field", "segment_id": segment_id_2,
                    "field": "template", "value": template_match.group(1).strip()
                })
            if evidence_match:
                evidence_text = evidence_match.group(1).strip()
                evidence_list_of_dicts = []
                raw_evidences = evidence_text.replace("\n", "").replace(";", "；").split("；")
                for e_str in raw_evidences:
                    e_str = e_str.strip()
                    if not e_str:
                        continue
                    is_static, static_value = False, None
                    if e_str.lower().endswith("(static)") or e_str.endswith("(静态)"):
                        is_static = True
                        e_str = re.sub(r'\s*\((static|静态)\)\s*$', '', e_str, flags=re.IGNORECASE).strip()
                    evidence_list_of_dicts.append({"text": e_str, "is_static": is_static, "value": static_value})

                instructions.append({
                    "type": "modify_segment_field",
                    "segment_id": segment_id_1,
                    "field": "evidences",
                    "value": evidence_list_of_dicts
                })
            print(f"    -> LLM 建议修正 Segment '{segment_id_2}'。")
            
        return instructions

    except Exception as e:
        print(f"    ! 在为跨 Segment 矛盾重规划时发生异常: {e}")
        return []


#  重规划指令生成器 (总入口)
async def generate_replan_instructions(
    classified_defects: Dict[str, List], 
    manuscript: Section, 
    llm_planner: ChatModelBase, 
    formatter: FormatterBase) -> List[Dict[str, Any]]:
    """
    总调度函数，根据分类后的缺陷，调用相应的重规划函数。
    """

    print("\n--- Re-planning Instruction Generator: 开始生成修正指令 ---")
    all_instructions = []

    # 处理“单点 Evidence 错误”
    if "single_evidence_error" in classified_defects:
        single_evidence_feedbacks = classified_defects["single_evidence_error"]
        print(f"  - 检测到 {len(single_evidence_feedbacks)} 条 '单点 Evidence 错误'，开始处理...")
        for feedback in single_evidence_feedbacks:
            instruction = await replan_for_single_evidence_error(feedback, manuscript, llm_planner, formatter)
            if instruction:
                all_instructions.append(instruction)

    # 处理“Writer 撰写错误” 
    elif "writer_error" in classified_defects:
        writer_error_feedbacks = classified_defects["writer_error"]
        print(f"  - 检测到 {len(writer_error_feedbacks)} 个 Segment 存在 'Writer 撰写错误'，开始处理...")
        for feedback in writer_error_feedbacks:
            # replan_for_writer_error 可能会返回多条指令，所以用 extend
            instructions = await replan_for_writer_error(feedback, manuscript, llm_planner, formatter)
            if instructions:
                all_instructions.extend(instructions)
    
    # 处理“区域性 Segment 错误”
    elif "regional_segment_error" in classified_defects:
        feedbacks = classified_defects["regional_segment_error"]
        print(f"  - 检测到 {len(feedbacks)} 个 Segment 存在 '区域性错误'，开始处理...")
        for feedback_group in feedbacks:
            instructions = await replan_for_regional_segment_error(feedback_group, manuscript, llm_planner, formatter)
            if instructions:
                all_instructions.extend(instructions)

    # 处理“宏观完整性缺失” 
    elif "macro_incompleteness" in classified_defects:
        feedbacks = classified_defects["macro_incompleteness"]
        print(f"  - 检测到 {len(feedbacks)} 条 '宏观完整性缺失' 反馈，开始处理...")
        for feedback in feedbacks:
            # replan_for_macro_incompleteness 可能会返回多条指令
            instructions = await replan_for_macro_incompleteness(feedback, manuscript, llm_planner, formatter)
            if instructions:
                all_instructions.extend(instructions)

    # 处理“跨 Segment 逻辑矛盾” 
    elif "cross_segment_contradiction" in classified_defects:
        feedbacks = classified_defects["cross_segment_contradiction"]
        print(f"  - 检测到 {len(feedbacks)} 条 '跨 Segment 矛盾' 反馈，开始处理...")
        for feedback in feedbacks:
            # replan_for_cross_segment_contradiction 可能会返回多条指令
            instructions = await replan_for_cross_segment_contradiction(feedback, manuscript, llm_planner, formatter)
            if instructions:
                all_instructions.extend(instructions)

    print(f"--- Re-planning Instruction Generator: 共生成 {len(all_instructions)} 条修正指令 ---")
    return all_instructions