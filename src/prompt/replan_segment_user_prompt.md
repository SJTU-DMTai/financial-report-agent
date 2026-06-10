当前任务：{task_desc}
当前写作日期：{cur_date}

当前 segment：
topic: {topic}
requirements:
{requirements}
template:
{template}

无法找到的 evidence：
{unavailable_evidences}

请输出 JSON：
{{
  "topic": "可选，修改后的 topic",
  "requirements": [
    "修改后的写作要求 1",
    "修改后的写作要求 2"
  ],
  "evidences": [
    {{
      "description": "替换后的证据 1",
      "entity": "证据主体",
      "aspect": "指标、事件或分析维度",
      "period": "时间范围",
      "scope": "口径范围",
      "required": true
    }},
    {{
      "description": "替换后的证据 2",
      "entity": "证据主体",
      "aspect": "指标、事件或分析维度",
      "period": "时间范围",
      "scope": "口径范围",
      "required": true
    }}
  ]
}}

requirements 必须是字符串数组，每一项是一条独立写作要求。
如果无法找到合理替代 evidence，可以输出空数组 `"evidences": []`，并**在 requirements 中删除或弱化对应论点**。
