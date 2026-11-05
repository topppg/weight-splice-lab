import pytest

from weight_splice_lab.harness.protocol import cut, parse_json_body, parse_reply


def test_single_python_block() -> None:
    r = parse_reply("I will look.\n\n```python\n# hypothesis\nx = 1\nx\n```\n")
    assert r.block is not None and r.block.tool == "python"
    assert r.block.body == "# hypothesis\nx = 1\nx"
    assert r.n_blocks == 1 and not r.unterminated


def test_first_of_several_blocks_wins() -> None:
    text = '```forward\n{"type": "generate"}\n```\nthen\n```python\nprint(1)\n```'
    r = parse_reply(text)
    assert r.block is not None and r.block.tool == "forward"
    assert r.n_blocks == 2


def test_other_fences_are_ignored() -> None:
    text = 'Example:\n```json\n{}\n```\n```bash\nls\n```\n```save_edit\n{"final": true}\n```'
    r = parse_reply(text)
    assert r.block is not None and r.block.tool == "save_edit" and r.n_blocks == 1


def test_no_block_and_unterminated() -> None:
    assert parse_reply("Thinking about it.").block is None
    r = parse_reply("```python\nprint(1)\n")
    assert r.block is None and r.unterminated


def test_inline_backticks_do_not_close_a_block() -> None:
    r = parse_reply('```python\ns = "```not a fence"\nprint(s)\n```')
    assert r.block is not None and r.block.body == 's = "```not a fence"\nprint(s)'


def test_indented_fence_and_trailing_spaces() -> None:
    r = parse_reply("  ```python  \nprint(2)\n  ```  ")
    assert r.block is not None and r.block.body == "print(2)"


def test_json_body() -> None:
    assert parse_json_body('{"final": true}') == {"final": True}
    assert parse_json_body("") == {}
    with pytest.raises(TypeError):
        parse_json_body("[1, 2]")
    with pytest.raises(ValueError):
        parse_json_body("{not json}")


def test_cut() -> None:
    assert cut("abc", 10) == "abc"
    out = cut("x" * 20, 10)
    assert out.startswith("x" * 10) and "cut at 10 characters" in out


def test_fence_inside_a_python_string_does_not_end_the_block() -> None:
    text = 'x\n```python\nnotes = """\n```\ncode\n```\n"""\nprint(1)\n```'
    r = parse_reply(text)
    assert r.block is not None and r.n_blocks == 1
    assert r.block.body == 'notes = """\n```\ncode\n```\n"""\nprint(1)'
    r = parse_reply("```python\nif 1:\n    s='''\n    ```\n    '''\n```")
    assert r.block is not None and r.block.body == "if 1:\n    s='''\n    ```\n    '''"


def test_broken_python_still_closes_at_the_first_fence() -> None:
    r = parse_reply('```python\ns = """\n```\nafter')
    assert r.block is not None and r.block.body == 's = """'


def test_py_alias_and_longer_fences() -> None:
    r = parse_reply("```py\nprint(1)\n```")
    assert r.block is not None and r.block.tool == "python"
    r = parse_reply("````python\nx = '''\n```\n'''\n````")
    assert r.block is not None and r.block.body == "x = '''\n```\n'''"
    assert parse_reply("````python\nprint(1)\n```").unterminated
