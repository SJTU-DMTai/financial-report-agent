# -*- coding: utf-8 -*-
"""The utilities for deep research agent"""
import os
import json
import re
from typing import Union, Sequence, Any, Type
from pydantic import BaseModel

from agentscope.tool import Toolkit, ToolResponse


TOOL_RESULTS_MAX_WORDS = 5000


def get_prompt_from_file(
    file_path: str,
    return_json: bool,
) -> Union[str, dict]:
    """Get prompt from file"""
    with open(os.path.join(file_path), "r", encoding="utf-8") as f:
        if return_json:
            prompt = json.load(f)
        else:
            prompt = f.read()
    return prompt


def truncate_by_words(sentence: str) -> str:
    """Truncate too long sentences by words number"""
    words = re.findall(
        r"\w+|[^\w\s]",
        sentence,
        re.UNICODE,
    )

    word_count = 0
    result = []
    for word in words:
        if re.match(r"\w+", word):
            word_count += 1
        if word_count > TOOL_RESULTS_MAX_WORDS:
            break
        result.append(word)

    truncated_sentence = ""
    for i, word in enumerate(result):
        if i == 0:
            truncated_sentence += word
        elif re.match(r"\w+", word):
            truncated_sentence += " " + word
        else:
            truncated_sentence += word
    return truncated_sentence


def truncate_search_result(
    res: list,
    search_func: str = "tavily-search",
    extract_function: str = "tavily-extract",
) -> list:
    """Truncate search result in deep research agent"""
    if search_func != "tavily-search" or extract_function != "tavily-extract":
        raise NotImplementedError(
            "Specific implementation of truncation should be provided.",
        )

    for i, val in enumerate(res):
        res[i]["text"] = truncate_by_words(val["text"])

    return res


def generate_structure_output(**kwargs: Any) -> ToolResponse:
    """Generate a structured output tool response.

    This function is designed to be used as a tool function for generating
    structured outputs. It takes arbitrary keyword arguments and wraps them
    in a ToolResponse with metadata.

    Args:
        **kwargs: Arbitrary keyword arguments that should match the format
            of the expected structured output specification.

    Returns:
        ToolResponse: A tool response object with empty content and the
            provided kwargs as metadata.

    Note:
        The input parameters should be in the same format as the specification
        and include as much detail as requested by the calling context.
    """
    return ToolResponse(content=[], metadata=kwargs)


def get_dynamic_tool_call_json(data_model_type: Type[BaseModel]) -> list[dict]:
    """Generate JSON schema for dynamic tool calling with a given data model.

    Creates a temporary toolkit, registers the structure output function,
    and configures it with the specified data model to generate appropriate
    JSON schemas for tool calling.

    Args:
        data_model_type: A Pydantic BaseModel class that defines the expected
            structure of the tool output.

    Returns:
        A list of dictionary that contains the JSON schemas for
        the configured tool, suitable for use in API calls that
        support structured outputs.

    Example:
        class MyModel(BaseModel):
            name: str
            value: int

        schema = get_dynamic_tool_call_json(MyModel)
    """
    tmp_toolkit = Toolkit()
    tmp_toolkit.register_tool_function(generate_structure_output)
    tmp_toolkit.set_extended_model(
        "generate_structure_output",
        data_model_type,
    )
    return tmp_toolkit.get_json_schemas()


def get_structure_output(blocks: list | Sequence) -> dict:
    """Extract structured output from a sequence of blocks.

    Processes a list or sequence of blocks to extract tool use outputs
    and combine them into a single dictionary. This is typically used
    to parse responses from language models that include tool calls.

    Args:
        blocks: A list or sequence of blocks that may contain tool use
            information. Each block should be a dictionary with 'type'
            and 'input' keys for tool use blocks.

    Returns:
        A dictionary containing the combined input data from all tool
        use blocks found in the input sequence.

    Example:
        blocks = [
            {"type": "tool_use", "input": {"name": "test"}},
            {"type": "text", "content": "Some text"},
            {"type": "tool_use", "input": {"value": 42}}
        ]
        result = PromptBase.get_structure_output(blocks)
        # result: {"name": "test", "value": 42}
    """

    dict_output = {}
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            dict_output.update(block.get("input", {}))
    return dict_output
