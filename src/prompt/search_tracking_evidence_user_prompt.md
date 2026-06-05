任务：{task_desc}
当前需要调研的 segment：{segment_topic}

已解决的相关证据：
{known_evidence}

当前证据需求：
{evidence_description}

参考研报片段：
{reference_context}

如果当前 evidence 并非可检索事实或可引用材料需求，例如画图要求、数据来源声明、格式要求，只输出 SKIP。
如果已解决证据或参考研报片段已经覆盖当前 evidence，必须摘取相关内容并保留其中的 cite_id，不要输出 SKIP。
否则请调用工具检索。最终答案必须简洁，事实、数字、新闻、公告和数据都必须带 [^cite_id:xxxxxx] 引用。
