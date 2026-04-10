from evals.utils import build_messages


def test_build_messages_uses_system_prompt():
    msgs = build_messages({"prompt": "hi"})
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_build_messages_custom_system_prompt():
    msgs = build_messages({"prompt": "hi", "system_prompt": "you are X"})
    assert msgs[0]["content"] == "you are X"
