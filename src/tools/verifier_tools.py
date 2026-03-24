# ../tools/verifier_tools.py
from __future__ import annotations
from typing import List, Union, Any
import re
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
from ..memory.short_term import ShortTermMemoryStore

class RefIdTools:
    def __init__(self, short_term: ShortTermMemoryStore):
        self.short_term = short_term

    def _extract_text_safe(self, tool_response: Any) -> str:
        """
        安全地提取文本内容
        """
        try:
            # 如果已经是字符串，直接返回
            if isinstance(tool_response, str):
                return tool_response
            
            # 如果是 ToolResponse 对象
            if hasattr(tool_response, 'content'):
                content = tool_response.content
                
                # 如果 content 是字符串
                if isinstance(content, str):
                    return content
                
                # 如果 content 是列表
                elif isinstance(content, list):
                    texts = []
                    for item in content:
                        # 如果是 TextBlock 对象
                        if hasattr(item, 'text'):
                            texts.append(str(item.text))
                        # 如果是字典
                        elif isinstance(item, dict):
                            # 尝试获取 text 字段
                            text = item.get('text') or item.get('content')
                            if text:
                                texts.append(str(text))
                        # 其他类型转换为字符串
                        else:
                            texts.append(str(item))
                    return "\n".join(texts)
                
                # 其他类型的 content
                else:
                    return str(content)
            
            # 如果是字典（可能来自 metadata）
            elif isinstance(tool_response, dict):
                # 尝试从常见字段中提取文本
                for field in ['text', 'content', 'output', 'result']:
                    if field in tool_response:
                        text = tool_response[field]
                        if isinstance(text, (str, int, float)):
                            return str(text)
                        elif isinstance(text, list):
                            return self._extract_text_safe(text)
            
            # 最后尝试直接转换为字符串
            return str(tool_response)
            
        except Exception as e:
            print(f"[ERROR _extract_text_safe] 提取文本失败: {str(e)}")
            return ""

    def extract_cite_ids(self, section_id: str) -> ToolResponse:
        """
        从章节正文中解析所有引用的 material cite_id
        """
        try:
            print(f"[DEBUG] 开始提取章节 {section_id} 的 cite_id")
            
            # 1. 通过 ManuscriptTools 读取章节内容
            from ..tools.manuscipt_tools import ManuscriptTools
            manuscript_tools = ManuscriptTools(self.short_term)
            
            # 2. 调用 read_manuscript_section 获取 ToolResponse
            section_result = manuscript_tools.read_manuscript_section(section_id)
            print(f"[DEBUG] section_result 类型: {type(section_result)}")
            
            # 3. 提取文本内容
            text = self._extract_text_safe(section_result)
            print(f"[DEBUG] 提取的文本长度: {len(text) if text else 0}")
            
            # 4. 打印文本前200字符以便调试
            if text and len(text) > 0:
                print(f"[DEBUG] 文本前200字符: {text[:200]}")
            
            if not text or text.strip() == "":
                return ToolResponse(
                    content=[TextBlock(type="text", text="章节内容为空")],
                    metadata={"section_id": section_id, "cite_ids": []}
                )
            
            # 5. 正则匹配 cite_id
            # 匹配 [^cite_id] 或 [^cite_id|描述] 格式
            pattern = re.compile(r'\[(?:\^cite_id:|\^|cite_id:)([^\|\]]+)(?:\|[^\]]*)?\]')
            matches = pattern.findall(text)
            print(f"[DEBUG] 找到匹配: {matches}")
            
            # 去重 + 保序
            seen = set()
            unique_cite_ids = []
            for rid in matches:
                if rid not in seen:
                    seen.add(rid)
                    unique_cite_ids.append(rid)
            
            print(f"[DEBUG] 去重后 cite_ids: {unique_cite_ids}")
            
            # 6. 准备输出
            if unique_cite_ids:
                result_text = f"在章节 {section_id} 中找到 {len(unique_cite_ids)} 个 cite_id:\n"
                for i, cite_id in enumerate(unique_cite_ids, 1):
                    result_text += f"{i}. {cite_id}\n"
            else:
                result_text = f"章节 {section_id} 中没有找到 cite_id 引用"
            
            return ToolResponse(
                content=[TextBlock(type="text", text=result_text)],
                metadata={
                    "section_id": section_id,
                    "cite_ids": unique_cite_ids,
                    "count": len(unique_cite_ids)
                }
            )
            
        except Exception as e:
            error_msg = f"提取 cite_id 时出错: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return ToolResponse(
                content=[TextBlock(type="text", text=error_msg)],
                metadata={"section_id": section_id, "error": str(e)}
            )


class SectionStatusTools:
    def __init__(self, short_term: ShortTermMemoryStore):
        self.short_term = short_term
    
    def check_section_status(self, section_id: str) -> ToolResponse:
        """
        检查章节状态：是否已完成撰写
        """
        try:
            print(f"[DEBUG SectionStatusTools.check_section_status] 检查章节 {section_id}")
            
            from ..tools.manuscipt_tools import ManuscriptTools
            manuscript_tools = ManuscriptTools(self.short_term)
            
            # 读取章节内容
            section_result = manuscript_tools.read_manuscript_section(section_id)
            
            # 使用安全的文本提取方法
            cite_id_tools = RefIdTools(self.short_term)
            text = cite_id_tools._extract_text_safe(section_result)
            
            print(f"[DEBUG] 检查章节内容长度: {len(text) if text else 0}")
            
            # 判断是否为大纲模板
            is_template = False
            template_indicators = [
                "请根据大纲要点在此撰写正文",
                "章节内容：",
                "写作风格和策略：",
                "字数范围："
            ]
            
            for indicator in template_indicators:
                if indicator in text:
                    is_template = True
                    break
            
            print(f"[DEBUG] 是否为模板: {is_template}")
            
            if is_template:
                return ToolResponse(
                    content=[TextBlock(
                        type="text", 
                        text=f"章节 {section_id} 尚未完成，仅包含大纲模板"
                    )],
                    metadata={
                        "section_id": section_id,
                        "status": "INCOMPLETE",
                        "is_template": True
                    }
                )
            else:
                # 统计字数
                count_result = manuscript_tools.count_manuscript_words(section_id)
                count_text = cite_id_tools._extract_text_safe(count_result)
                
                return ToolResponse(
                    content=[TextBlock(
                        type="text",
                        text=f"章节 {section_id} 已完成撰写\n{count_text}"
                    )],
                    metadata={
                        "section_id": section_id,
                        "status": "COMPLETE",
                        "is_template": False
                    }
                )
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"检查章节状态时出错: {str(e)}"
            print(f"[ERROR SectionStatusTools.check_section_status] {error_msg}")
            print(f"[ERROR] 详细错误: {error_details}")
            
            return ToolResponse(
                content=[TextBlock(type="text", text=error_msg)],
                metadata={"section_id": section_id, "error": str(e)}
            )