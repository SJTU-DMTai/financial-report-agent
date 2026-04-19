# -*- coding: utf-8 -*-
import json
import re
import traceback
import warnings
from collections import Counter

import pandas as pd
from agentscope.formatter import OpenAIChatFormatter
from abc import abstractmethod
from typing import Any, List, Type

from agentscope.formatter import OpenAIChatFormatter
from agentscope._utils._common import _save_base64_data, _json_loads_with_repair
from agentscope.formatter._openai_formatter import _to_openai_image_url, _to_openai_audio_data, logger
from agentscope.message import Msg, AudioBlock, ImageBlock, TextBlock, ToolUseBlock, Base64Source, ThinkingBlock
from datetime import datetime
from src.memory.working import Section
from agentscope.model import OpenAIChatModel, ChatResponse
from agentscope.model._model_usage import ChatUsage
from openai import BaseModel
from openai.types.chat import ChatCompletion



def fmt_yyyymmdd(s: str) -> str:
    """
    返回YYYY-MM-DD
    """
    s = (s or "").strip()
    if not s:
        return s
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception as e:
        warnings.warn(f"DATE_FORMAT_ERROR: {s}")
        return s


class KaLMChatModel(OpenAIChatModel):

    def _parse_openai_completion_response(
        self,
        start_datetime: datetime,
        response: ChatCompletion,
        structured_model: Type[BaseModel] | None = None,
    ) -> ChatResponse:
        """Given an OpenAI chat completion response object, extract the content
            blocks and usages from it.

        Args:
            start_datetime (`datetime`):
                The start datetime of the response generation.
            response (`ChatCompletion`):
                OpenAI ChatCompletion object to parse.
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output.

        Returns:
            ChatResponse (`ChatResponse`):
                A ChatResponse object containing the content blocks and usage.

        .. note::
            If `structured_model` is not `None`, the expected structured output
            will be stored in the metadata of the `ChatResponse`.
        """
        content_blocks: List[
            TextBlock | ToolUseBlock | ThinkingBlock | AudioBlock
        ] = []
        metadata: dict | None = None

        if response.choices:
            choice = response.choices[0]
            if (
                hasattr(choice.message, "reasoning_content")
                and choice.message.reasoning_content is not None
            ):
                content_blocks.append(
                    ThinkingBlock(
                        type="thinking",
                        thinking=response.choices[0].message.reasoning_content,
                    ),
                )

            if choice.message.content:
                if choice.message.content.startswith("<think>") and "</think>" in choice.message.content:
                    s = re.search("<think>(.*)</think>(.*)", choice.message.content, re.DOTALL)
                    content_blocks.append(
                        ThinkingBlock(
                            type="thinking",
                            thinking=s.group(1).strip(),
                        )
                    )
                    if s.group(2).strip():
                        content_blocks.append(
                            TextBlock(
                                type="text",
                                text=s.group(2).strip(),
                            )
                        )
                else:
                    content_blocks.append(
                        TextBlock(
                            type="text",
                            text=response.choices[0].message.content,
                        ),
                    )
            if choice.message.audio:
                media_type = self.generate_kwargs.get("audio", {}).get(
                    "format",
                    "mp3",
                )
                content_blocks.append(
                    AudioBlock(
                        type="audio",
                        source=Base64Source(
                            data=choice.message.audio.data,
                            media_type=f"audio/{media_type}",
                            type="base64",
                        ),
                    ),
                )

                if choice.message.audio.transcript:
                    content_blocks.append(
                        TextBlock(
                            type="text",
                            text=choice.message.audio.transcript,
                        ),
                    )

            for tool_call in choice.message.tool_calls or []:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=_json_loads_with_repair(
                            tool_call.function.arguments,
                        ),
                    ),
                )

            if structured_model:
                metadata = choice.message.parsed.model_dump()

        usage = None
        if response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                time=(datetime.now() - start_datetime).total_seconds(),
                metadata=response.usage,
            )

        parsed_response = ChatResponse(
            content=content_blocks,
            usage=usage,
            metadata=metadata,
        )

        return parsed_response


_SECTION_NUMBER_PREFIX_PATTERNS = (
    re.compile(r"^\d+(?:\.\d+)+\s+"),
    re.compile(r"^\d+\.\s+"),
    re.compile(r"^\d+\s+(?!年\b|月\b|日\b)"),
    re.compile(r"^[一二三四五六七八九十百千]+[、.]\s*"),
    re.compile(r"^[（(]\d+[）)]\s*"),
    re.compile(r"^[（(][一二三四五六七八九十百千]+[）)]\s*"),
)


def _strip_section_number_prefix(title: str) -> str:
    text = (title or "").strip()
    for pattern in _SECTION_NUMBER_PREFIX_PATTERNS:
        updated = pattern.sub("", text, count=1).strip()
        if updated != text:
            return updated
    return text


def _normalize_section_titles(section: Section) -> None:
    section.title = _strip_section_number_prefix(section.title)
    for subsection in section.subsections or []:
        _normalize_section_titles(subsection)


def _has_renderable_content(section: Section) -> bool:
    if getattr(section, "content", None) and section.content.strip():
        return True
    for seg in section.segments or []:
        if getattr(seg, "content", None) and seg.content.strip():
            return True
    for subsection in section.subsections or []:
        if _has_renderable_content(subsection):
            return True
    return False


def _count_renderable_top_level_sections(manuscript: Section) -> int:
    return sum(1 for sub in (manuscript.subsections or []) if _has_renderable_content(sub))


def _infer_report_title(task_desc: str, entity: dict) -> str:
    company_name = str(entity.get("name", "")).strip()
    stock_code = str(entity.get("code", "")).strip()
    suffix = "深度研究报告"
    for candidate in ("深度研究报告", "研究报告", "年报点评", "季报点评"):
        if candidate in (task_desc or ""):
            suffix = candidate
            break
    return f"{company_name}（{stock_code}）{suffix}".strip()


def _normalize_report_title(manuscript: Section, entity: dict, task_desc: str) -> None:
    manuscript.title = _strip_section_number_prefix(manuscript.title)
    if _count_renderable_top_level_sections(manuscript) == 1:
        manuscript.title = _infer_report_title(task_desc, entity)

def extract_tagged_text(text: str, tag: str = "content") -> str | None:
    if not text:
        return None

    pattern = re.compile(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def extract_writer_content(text: str) -> str:
    tagged_text = extract_tagged_text(text, "content")
    if tagged_text is not None:
        return tagged_text
    return (text or "").strip()


def extract_cite_ids(text: str) -> list[str]:
    if not text:
        return []
    pattern = re.compile(r"\[\^cite_id[:=]\s*([^\]]+?)\s*\]")
    return [m.group(1).strip() for m in pattern.finditer(text)]


def extract_chart_ids(text: str) -> list[str]:
    if not text:
        return []
    pattern = re.compile(
        r'!\[(?P<alt>.*?)\]\(chart:(?P<chart_id>[a-zA-Z0-9_\-]+)\)'
    )
    return [m.group("chart_id").strip() for m in pattern.finditer(text)]


def print_section_reference_warning(section_title: str, before_text: str, after_text: str) -> None:
    before_cites = Counter(extract_cite_ids(before_text))
    after_cites = Counter(extract_cite_ids(after_text))
    before_charts = Counter(extract_chart_ids(before_text))
    after_charts = Counter(extract_chart_ids(after_text))

    missing_cites = []
    for cite_id, count in before_cites.items():
        if after_cites[cite_id] < count:
            missing_cites.append(f"{cite_id} x{count - after_cites[cite_id]}")

    missing_charts = []
    for chart_id, count in before_charts.items():
        if after_charts[chart_id] < count:
            missing_charts.append(f"{chart_id} x{count - after_charts[chart_id]}")

    if not missing_cites and not missing_charts:
        return

    print("\n[WARN] 章节润色后检测到引用或图表被删去", flush=True)
    print(f"[WARN] 章节: {section_title}", flush=True)
    if missing_cites:
        print(f"[WARN] 缺失 cite_id: {missing_cites}", flush=True)
    if missing_charts:
        print(f"[WARN] 缺失 chart_id: {missing_charts}", flush=True)
    print("[WARN] 修改前内容:", flush=True)
    print(before_text, flush=True)
    print("[WARN] 修改后内容:", flush=True)
    print(after_text, flush=True)


class PatchedOpenAIChatFormatter(OpenAIChatFormatter):
    """
    在原有 OpenAIChatFormatter 基础上，对 tool_result 里的
    unsupported block（比如 type='thinking'）做兼容处理，
    只保留 text/image/audio/video，其它跳过。
    防止出现thinking block报错
    """
    @staticmethod
    def convert_tool_result_to_string(output) -> tuple:
        """Turn the tool result list into a textual output to be compatible
        with the LLM API that doesn't support multimodal data in the tool
        result.

        For URL-based images, the URL is included in the list. For
        base64-encoded images, the local file path where the image is saved
        is included in the returned list.

        Args:
            output (`str | List[TextBlock | ImageBlock | AudioBlock | \
            VideoBlock]`):
                The output of the tool response, including text and multimodal
                data like images and audio.

        Returns:
            `tuple[str, list[Tuple[str, ImageBlock | AudioBlock | VideoBlock \
            TextBlock]]]`:
                A tuple containing the textual representation of the tool
                result and a list of tuples. The first element of each tuple
                is the local file path or URL of the multimodal data, and the
                second element is the corresponding block.
        """

        if isinstance(output, str):
            return output, []

        textual_output = []
        multimodal_data = []
        for block in output:
            assert isinstance(block, dict) and "type" in block, (
                f"Invalid block: {block}, a TextBlock, ImageBlock, "
                f"AudioBlock, or VideoBlock is expected."
            )
            if block["type"] == "text":
                textual_output.append(block["text"])

            elif block["type"] in ["image", "audio", "video"]:
                assert "source" in block, (
                    f"Invalid {block['type']} block: {block}, 'source' key "
                    "is required."
                )
                source = block["source"]
                # Save the image locally and return the file path
                if source["type"] == "url":
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {source['url']}",
                    )

                    path_multimodal_file = source["url"]

                elif source["type"] == "base64":
                    path_multimodal_file = _save_base64_data(
                        source["media_type"],
                        source["data"],
                    )
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {path_multimodal_file}",
                    )

                else:
                    raise ValueError(
                        f"Invalid image source: {block['source']}, "
                        "expected 'url' or 'base64'.",
                    )

                multimodal_data.append(
                    (path_multimodal_file, block),
                )

            else:
                # raise ValueError(
                #     f"Unsupported block type: {block['type']}, "
                #     "expected 'text', 'image', 'audio', or 'video'.",
                # )
                continue

        if len(textual_output) == 1:
            return textual_output[0], multimodal_data

        else:
            return "\n".join("- " + _ for _ in textual_output), multimodal_data
