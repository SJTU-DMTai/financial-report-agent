# 任务
请基于你上一轮已经指出的问题和修改方向，输出**最终的 JSON 修改方案**。

你的输出会被程序直接解析并执行，因此必须保证：
1. JSON 结构严格正确；
2. 字段含义使用准确；
3. 新生成的 segment 信息能够被后续 evidence 搜索和写作流程直接使用。

## 术语说明
- `outline`：整份研报的结构化大纲，由多层 `section` 和 `segment` 组成。
- `section`：章节节点，对应研报中的某个章节或子章节，主要作用是组织结构。一个 `section` 可以包含多个 `segment`，也可以包含下级 `subsection`。
- `segment`：章节中的一个写作单元。后续 evidence search 和 writer 通常会以 `segment` 为粒度执行，所以一个 `segment` 应尽量只围绕一个主题。
- `topic`：该 `segment` 的主题标签，说明该写作单元主要分析什么。
- `requirements`：该 `segment` 的写作要求，描述后续正文需要覆盖的内容要素、分析角度、结构方式或呈现重点。
- `template`：历史研报抽象出的示例写法，只是参考性的写作顺序或表达框架，不是必须字段。对新生成的 `segment`，不需要提供 `template`。
- `evidence` / `evidences`：后续要搜索和核实的论据项列表。每条 `evidence` 都应该能直接服务于对应 `segment`，并尽量做到可搜索、指代明确。
- `modify / add / delete / split / merge`：分别表示修改已有节点、增加新节点、删除低价值节点、拆分主题混杂的节点、合并语义高度重合的相邻节点。

## 总体要求
1. 只输出固定格式 JSON，不要输出 Markdown、解释、代码块围栏或额外说明。
2. 只做必要修改。能通过一次 `modify_segment` 解决的问题，就不要拆成多条操作。
3. 如果无需修改，输出：
```json
{
  "operations": []
}
```
4. 所有已有对象的 id 必须来自输入 outline 中已有的 `SEC_xxx` / `SEG_xxx`。
5. 不要让后面的操作依赖前面新创建出来的 section 或 segment。所有操作都必须只引用原 outline 中已有的 id。
6. 新增的 section 或 segment 不需要你生成新的 id，程序会在执行时分配。

## 输出 JSON 字段
- 顶层只保留一个字段：`operations`
- `operations` 是一个数组，表示要执行的修改操作列表。每个元素都是一条独立操作。

## 与位置相关的字段
- `section_id`：指向一个已有 section，表示这条操作针对哪个已有 section。
- `segment_id`：指向一个已有 segment，表示这条操作针对哪个已有 segment。
- `parent_section_id`：仅用于 `add_section`，表示新 section 要被插入到哪个已有父 section 下面。
- `insert_index`：表示插入位置的顺序索引，从 0 开始。
  - 对 `add_section`：表示新 section 插入到父 section 的第几个子 section 位置。
  - 对 `add_segment`：表示新 segment 插入到该 section 的第几个 segment 位置。
  如果你无法确定最合适的位置，可以省略该字段，让程序按默认位置追加。

## 与内容更新相关的字段
- `updates`：仅用于 `modify_section` 或 `modify_segment`，表示对原对象的局部更新。它只应包含需要改动的字段，不要把没改的字段重复填进去。
- `new_section`：仅用于 `add_section`，表示要新增的 section 内容。
- `new_segment`：用于 `add_segment` 或 `merge_segments`，表示新增出来的 segment 内容。
- `new_segments`：仅用于 `split_segment`，表示把原 segment 拆分后得到的多个新 segment。

## segment 内容字段
无论出现在 `updates`、`new_segment`、`new_section.segments[*]` 还是 `new_segments[*]` 中，下面这些字段都表示同样的含义：

- `topic`：
  该 segment 未来要写什么。应单一、明确、可作为一个独立写作单元。不要写成过宽的标题，也不要混入两个并列主题。

- `requirements`：
  该 segment 未来写作时必须覆盖什么。应写清分析重点、论证方向、信息边界。它不是泛泛的“展开分析”，而应让写作者知道这一段到底要完成什么任务。

- `evidences`：
  后续系统需要搜索和核实的论据项列表。每条 evidence 都应尽量可检索、指代明确，并能直接支持该 segment的写作。
  不要写成空泛口号，例如“相关资料”“公司情况”“行业信息”，或者是无意义信息。
  优先写成带明确对象和维度的表述，例如“公司近三年收入与归母净利润变化”“主要竞争对手及市场份额比较”。
  evidence 数量应该在6条以内，避免加大搜索负担。

## evidence 字段
每条 evidence 是一个对象，只包含：

- `text`：
  这条论据要搜索什么。应尽量具体、清楚、可检索。

## 允许的 action
- `modify_section`
- `add_section`
- `delete_section`
- `modify_segment`
- `add_segment`
- `delete_segment`
- `merge_segments`
- `split_segment`

## 各 action 的具体要求

### 1. modify_section
适用场景：
- section 标题不准确；
- section 职责需要小幅调整；
- 不需要新增或删除整个 section，只需修改已有 section。

必填字段：
- `section_id`
- `updates.title`

注意：
- 这里只改 section 本身，不改其下具体 segments。
- 如果问题在于 section 内部某个 segment，不要误用 `modify_section`。

### 2. add_section
适用场景：
- 当前 outline 缺少一个必要章节，且这个章节不适合仅通过新增单个 segment 解决。

必填字段：
- `parent_section_id`
- `new_section`

可选字段：
- `insert_index`

`new_section` 必须包含：
- `title`
- `segments`

`new_section.segments[*]` 中每个 segment 必须包含：
- `topic`
- `requirements`
- `evidences`

注意：
- `new_section.segments[*]` 不要提供 `template`。
- 只有在确实需要新增一个完整章节时才使用 `add_section`。如果只是补一个分析点，优先使用 `add_segment`。

### 3. delete_section
适用场景：
- 某个 section 明显低价值、与任务不相关、或与其他 section 高度重复，适合整体删除。

必填字段：
- `section_id`

注意：
- 只有在整个章节都应移除时才使用。
- 若只是 section 内某个 segment 有问题，不要删除整个 section。

### 4. modify_segment
适用场景：
- 原 segment 主题还值得保留，但 `topic`、`requirements`、`template` 或 `evidences` 需要调整。

必填字段：
- `segment_id`
- `updates`

`updates` 可包含：
- `topic`
- `requirements`
- `template`
- `evidences`

注意：
- `modify_segment` 用于“保留这个 segment，但把它改得更合理”。
- 如果原 segment 应被拆成多个主题，请改用 `split_segment`。
- 如果原 segment 应与相邻 segment 合并，请改用 `merge_segments`。

### 5. add_segment
适用场景：
- 当前 section 下缺少一个必要分析单元，但不需要新开一个 section。

必填字段：
- `section_id`
- `new_segment`

可选字段：
- `insert_index`

`new_segment` 必须包含：
- `topic`
- `requirements`
- `evidences`

注意：
- `new_segment` 不要提供 `template`。
- 新增 segment 时，必须保证它承担的是一个清晰、独立的结构职责。

### 6. delete_segment
适用场景：
- 某个 segment 低价值、重复、与任务无关，且不需要保留其内容。

必填字段：
- `segment_id`

注意：
- 仅当删除后不会破坏该 section 的核心结构时才使用。
- 如果这个 segment 的有价值部分可以通过修改保留，优先不要删除。

### 7. merge_segments
适用场景：
- 两个或多个相邻 segments 主题高度重合，分开写会重复展开或职责不清。

必填字段：
- `segment_ids`
- `new_segment`

要求：
- `segment_ids` 必须指向同一 section 下的相邻 segments。
- `new_segment` 必须包含：
  - `topic`
  - `requirements`
  - `evidences`

注意：
- `new_segment` 不要提供 `template`。
- 合并后得到的新 segment 应保留原 segments 的有效信息，但主题要更聚焦，evidences 也应去重和收敛。

### 8. split_segment
适用场景：
- 原 segment 混入多个主题，导致后续检索和写作发散，必须拆开。

必填字段：
- `segment_id`
- `new_segments`

要求：
- `new_segments` 至少 2 个。
- 每个新 segment 都必须包含：
  - `topic`
  - `requirements`
  - `evidences`

注意：
- 每个新 segment 都不要提供 `template`。
- 拆分后的每个新 segment 应只围绕一个较单一的主题展开。

## 生成原则
1. 优先保留已有 outline 中合理的部分，只修改那些确实存在明显问题的section 或 segment。
2. 新增或改写的 `topic` 必须单一明确。
3. `requirements` 必须能指导后续写作，不能空泛。
4. `evidences` 必须便于后续检索，避免过泛、不可检索或缺少对象指代。
5. 不要为了追求“创新”而加入难以支撑、难以检索或明显超出任务范围的新 segment。

## 输出格式示例
{
  "operations": [
    {
      "action": "modify_segment",
      "segment_id": "SEG_003",
      "updates": {
        "topic": "行业竞争格局与公司所处位置",
        "requirements": "说明公司所在行业的竞争态势、主要参与者以及公司所处位置，避免展开到无关的产业链分析。",
        "evidences": [
          {
            "text": "公司所在细分行业的主要竞争者及竞争格局"
          },
          {
            "text": "公司在行业中的市场地位或竞争优势依据"
          }
        ]
      }
    },
    {
      "action": "add_segment",
      "section_id": "SEC_002",
      "insert_index": 1,
      "new_segment": {
        "topic": "未来一年的潜在催化因素",
        "requirements": "识别未来一年可能影响公司基本面或估值重估的关键催化因素，并说明其影响路径。",
        "evidences": [
          {
            "text": "未来一年公司潜在催化因素及时间线"
          },
          {
            "text": "各催化因素可能影响收入、利润或估值的路径"
          }
        ]
      }
    }
  ]
}
