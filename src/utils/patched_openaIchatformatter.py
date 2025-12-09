# -*- coding: utf-8 -*-
from agentscope.formatter import OpenAIChatFormatter
from abc import abstractmethod
from typing import Any, List

from agentscope.formatter import OpenAIChatFormatter
from agentscope._utils._common import _save_base64_data
from agentscope.message import Msg, AudioBlock, ImageBlock, TextBlock

class PatchedOpenAIChatFormatter(OpenAIChatFormatter):
    """
    在原有 OpenAIChatFormatter 基础上，对 tool_result 里的
    unsupported block（比如 type='thinking'）做兼容处理，
    只保留 text/image/audio/video，其它跳过。
    防止出现thinking block报错
    """
    @staticmethod
    def convert_tool_result_to_string(
    output: str | List[TextBlock | ImageBlock | AudioBlock],
) -> str:
        """Turn the tool result list into a textual output to be compatible
        with the LLM API that doesn't support multimodal data.

        Args:
            output (`str | List[TextBlock | ImageBlock | AudioBlock]`):
                The output of the tool response, including text and multimodal
                data like images and audio.

        Returns:
            `str`:
                A string representation of the tool result, with text blocks
                concatenated and multimodal data represented by file paths
                or URLs.
        """

        if isinstance(output, str):
            return output

        textual_output = []
        for block in output:
            assert isinstance(block, dict) and "type" in block, (
                f"Invalid block: {block}, a TextBlock, ImageBlock, or "
                f"AudioBlock is expected."
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

                elif source["type"] == "base64":
                    path_temp_file = _save_base64_data(
                        source["media_type"],
                        source["data"],
                    )
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {path_temp_file}",
                    )

                else:
                    raise ValueError(
                        f"Invalid image source: {block['source']}, "
                        "expected 'url' or 'base64'.",
                    )

            else:
                # raise ValueError(
                #     f"Unsupported block type: {block['type']}, "
                #     "expected 'text', 'image', 'audio', or 'video'.",
                # )

                # 任何未知类型都直接忽略
                continue

        if len(textual_output) == 1:
            return textual_output[0]

        else:
            return "\n".join("- " + _ for _ in textual_output)