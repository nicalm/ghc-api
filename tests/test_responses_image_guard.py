"""Image counting uses content blocks without interpreting tool arguments."""

import copy

import pytest

from ghc_api.responses_image_guard import (
    count_input_images,
    has_input_images,
    image_limit_error,
)


def picture(label="synthetic"):
    return {"type": "input_image", "image_url": "data:image/png;base64," + label}


def test_typed_tool_output_images_exceed_limit_without_counting_schema():
    payload = {
        "model": "gpt-test",
        "input": [
            {"type": "additional_tools", "tools": [
                {"type": "function", "parameters": {"default": picture()}}
            ]},
            {"type": "custom_tool_call", "call_id": "synthetic-call",
             "input": '{"type":"input_image"}'},
            {"type": "custom_tool_call_output", "call_id": "synthetic-call", "output": [
                {"type": "input_text", "text": "Synthetic image batch"},
                *[picture(str(index)) for index in range(52)],
            ]},
        ],
    }
    snapshot = copy.deepcopy(payload)
    assert count_input_images(payload) == 52
    assert has_input_images(payload)
    error = image_limit_error(payload)
    assert error["error"]["code"] == "context_length_exceeded"
    assert error["error"]["param"] == "input"
    assert "52 images (maximum 50)" in error["error"]["message"]
    assert payload == snapshot


def test_exact_limit_is_forwardable_and_duplicate_blocks_still_count():
    payload = {"input": [{"role": "user", "content": [picture() for _ in range(50)]}]}
    assert count_input_images(payload) == 50
    assert has_input_images(payload)
    assert image_limit_error(payload) is None


def test_message_function_output_and_direct_image_parts_are_combined():
    payload = {"input": [
        {"type": "message", "role": "user", "content": [picture()]},
        {"type": "function_call_output", "call_id": "synthetic-call", "output": [
            picture(), picture()
        ]},
        picture(),
    ]}
    assert count_input_images(payload) == 4
    assert has_input_images(payload)


def test_schema_defaults_metadata_and_opaque_strings_do_not_count():
    payload = {"metadata": {"image": picture()}, "input": [
        {"type": "additional_tools", "tools": [{"parameters": {"example": picture()}}]},
        {"type": "function_call", "arguments": '{"image":{"type":"input_image"}}'},
        {"role": "user", "content": [{"type": "input_text", "text": "type: input_image"}]},
        {"type": "function_call_output", "output": '{"type":"input_image"}'},
        {"type": "reasoning", "content": [picture()]},
    ]}
    assert count_input_images(payload) == 0
    assert not has_input_images(payload)
    assert image_limit_error(payload) is None


@pytest.mark.parametrize("payload", [{"input": "Please describe an input_image"}, {}, {"input": []}])
def test_text_only_and_empty_input(payload):
    assert not has_input_images(payload)
    assert image_limit_error(payload) is None


def test_malformed_type_does_not_crash_preflight():
    payload = {"input": [{"type": {"invalid": True}}, {"type": ["message"]}, None, 1]}
    assert count_input_images(payload) == 0
    assert image_limit_error(payload) is None


def test_custom_limit():
    assert image_limit_error({"input": [picture(), picture()]}, max_images=1) is not None


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "50", None])
def test_limit_must_be_a_positive_integer(invalid):
    with pytest.raises(ValueError):
        image_limit_error({"input": []}, max_images=invalid)
