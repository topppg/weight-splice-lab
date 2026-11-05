import io
import json
import re
import tokenize
from dataclasses import dataclass

TOOLS = ("python", "forward", "save_edit")
ALIASES = {"py": "python"}
_OPEN = re.compile(r"^(`{3,})(" + "|".join((*TOOLS, *ALIASES)) + r")[ \t]*$")
_CLOSE = re.compile(r"^(`{3,})[ \t]*$")


@dataclass(frozen=True)
class ToolBlock:
    tool: str
    body: str


@dataclass(frozen=True)
class ParsedReply:
    block: ToolBlock | None
    n_blocks: int
    unterminated: bool


def _inside_string(code: str) -> bool:
    try:
        for _ in tokenize.generate_tokens(io.StringIO(code + "\n").readline):
            pass
    except tokenize.TokenError as e:
        return "multi-line string" in str(e.args[0])
    except SyntaxError:
        return False
    return False


def _close_at(lines: list[str], start: int, fence: int, tool: str) -> int | None:
    closes = [
        j for j in range(start, len(lines))
        if (m := _CLOSE.match(lines[j].strip())) and len(m.group(1)) >= fence
    ]  # fmt: skip
    if not closes:
        return None
    if tool == "python":
        # a fence line inside a triple-quoted string does not end the block
        for j in closes:
            if not _inside_string("\n".join(lines[start:j])):
                return j
    return closes[0]


def parse_reply(text: str) -> ParsedReply:
    blocks: list[ToolBlock] = []
    lines = text.splitlines()
    i = 0
    unterminated = False
    while i < len(lines):
        m = _OPEN.match(lines[i].strip())
        if m is None:
            i += 1
            continue
        tool = ALIASES.get(m.group(2), m.group(2))
        j = _close_at(lines, i + 1, len(m.group(1)), tool)
        if j is None:
            unterminated = True
            break
        blocks.append(ToolBlock(tool, "\n".join(lines[i + 1 : j])))
        i = j + 1
    return ParsedReply(blocks[0] if blocks else None, len(blocks), unterminated)


def parse_json_body(body: str) -> dict[str, object]:
    data = json.loads(body) if body.strip() else {}
    if not isinstance(data, dict):
        raise TypeError("the block must contain one JSON object")
    return data


def cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[output cut at {limit:,} characters]"
