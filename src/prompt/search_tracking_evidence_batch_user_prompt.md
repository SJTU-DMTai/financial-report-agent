任务：{task_desc}
当前需要调研的 segment：{segment_topic}

{reference_context}

当前有一组可并行检索的 evidence，请调用工具进行检索，并按 evidence_id 分别输出结果。

字段说明：
- evidence_id：当前 evidence 的唯一编号，最终 XML 必须原样返回。
- description：当前需要检索或确认的事实、数据或事件。
- known_evidence：可能对检索当前 evidence 有帮助的信息，但可能为空；如果已经足够覆盖 description，可以直接基于它作答并保留 cite_id。

{evidences_xml}

处理规则：
1. 如果某个 evidence 并非可检索事实或可引用材料需求，例如画图要求、数据来源声明、格式要求，该 evidence 输出 SKIPPED。
2. 如果某个 evidence 的 known_evidence 或参考研报片段已经覆盖当前 evidence，摘取相关内容并保留其中的 cite_id，不要输出 SKIPPED。
3. 否则请调用工具检索。可以为同批相近 evidence 复用同一次搜索结果，但每个 evidence 的 answer 必须分别对应其 description。
4. 最终 answer 必须简洁，只保留关键事实、数字、事件、公告或数据，并带 [^cite_id:xxxxxx] 引用。
5. 如果已经充分检索，但仍找不到可引用材料和确切搜索结果，该 evidence 输出 UNAVAILABLE，不要解释。
6. 每个输入 evidence_id 都必须有且只有一个 evidence_result；不要遗漏，也不要新增 evidence_id。

最终只输出 XML，不要输出 Markdown 或解释文字。XML 格式如下：
<results>
  <evidence_result>
    <evidence_id>ev_001</evidence_id>
    <status>RESOLVED|SKIPPED|UNAVAILABLE</status>
    <answer>简洁检索结果摘录；RESOLVED 时必须包含 [^cite_id:xxxxxx]</answer>
  </evidence_result>
  <evidence_result>
    <evidence_id>ev_002</evidence_id>
    <status>RESOLVED|SKIPPED|UNAVAILABLE</status>
    <answer>简洁检索结果摘录；RESOLVED 时必须包含 [^cite_id:xxxxxx]</answer>
  </evidence_result>
</results>
