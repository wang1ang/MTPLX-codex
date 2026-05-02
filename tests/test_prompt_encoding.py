from __future__ import annotations

from mtplx.benchmarks.schema import PromptCase, encode_prompt_case


class DummyTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        assert kwargs in ({}, {"enable_thinking": False})
        rendered = "".join(f"{m['role']}:{m['content']}\n" for m in messages) + "assistant:"
        return [ord(ch) for ch in rendered]


def test_encode_prompt_case_raw():
    case = PromptCase(id="x", category="raw", prompt="abc")
    assert encode_prompt_case(DummyTokenizer(), case, chat_template=False) == [97, 98, 99]


def test_encode_prompt_case_chat_template_wraps_user_prompt():
    case = PromptCase(id="x", category="chat", prompt="hello")
    encoded = encode_prompt_case(DummyTokenizer(), case, chat_template=True)
    assert encoded[:5] == [117, 115, 101, 114, 58]


def test_encode_prompt_case_uses_explicit_messages():
    case = PromptCase(
        id="x",
        category="chat",
        prompt="ignored",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    )
    encoded = encode_prompt_case(DummyTokenizer(), case, chat_template=True)
    assert chr(encoded[0]) == "s"


def test_encode_prompt_case_can_disable_thinking():
    case = PromptCase(id="x", category="chat", prompt="hello")
    encoded = encode_prompt_case(DummyTokenizer(), case, chat_template=True, enable_thinking=False)
    assert encoded[:5] == [117, 115, 101, 114, 58]
