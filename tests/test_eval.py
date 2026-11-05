import pytest

from weight_splice_lab.eval import (
    Problem,
    build_user_message,
    extract_answer,
    grade,
    is_correct,
    last_boxed,
    summarize,
    tail_duplicate_share,
    verbatim_loop_end,
)


def test_build_user_message() -> None:
    msg = build_user_message("What is 1+1?")
    suffix = "Please reason step by step, and put your final answer within \\boxed{}."
    assert msg == "What is 1+1?\n" + suffix


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so \\boxed{3} then \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("\\boxed{ {a}+{b} }", "{a}+{b}"),
        ("\\fbox{7}", "7"),
        ("answer \\boxed 5$ done", "5"),
        ("cut inside \\boxed{\\frac{1}{", None),
        ("empty \\boxed{}", None),
        ("no box here", None),
    ],
)
def test_last_boxed(text: str, expected: str | None) -> None:
    assert last_boxed(text) == expected


def test_extract_answer_reads_after_think() -> None:
    text = "<think>maybe \\boxed{1}</think>\nFinal: \\boxed{2}"
    assert extract_answer(text) == "2"
    assert extract_answer("thinking \\boxed{4} never closed") == "4"
    assert extract_answer("<think>\\boxed{1}</think> no box after") is None


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("\\frac{1}{2}", "0.5", True),
        ("\\dfrac{3}{4}", "\\frac34", True),
        ("2\\sqrt{2}", "\\sqrt{8}", True),
        ("(1,2)", "(1, 2)", True),
        ("3", "4", False),
        (None, "4", False),
    ],
)
def test_is_correct(pred: str | None, gold: str, ok: bool) -> None:
    assert is_correct(pred, gold) is ok


def test_loop_signals() -> None:
    phrase = "Wait, if he spends 1 hour walking, he cannot reach the end. "
    loop = "intro words here " + phrase * 20 + "Wai"
    assert verbatim_loop_end(loop)
    assert tail_duplicate_share(loop) > 0.9
    prose = " ".join(f"w{i}" for i in range(300))
    assert not verbatim_loop_end(prose)
    assert tail_duplicate_share(prose) == 0.0


def test_grade_and_summarize() -> None:
    p = Problem(id="x", problem="1+1", answer="2")
    good = grade(p, 0, "<think>easy</think> \\boxed{2}", tokens=10, finish="stop")
    cut = grade(p, 0, "Wait, check again. " * 50, tokens=4096, finish="length")
    assert good.correct and good.closed_think and not good.cut
    assert not cut.correct and cut.cut and cut.verbatim_loop
    s = summarize([good, cut])
    assert s["acc"] == 0.5
    assert s["cut_share"] == 0.5
    assert s["cut_verbatim_loop_share"] == 1.0
    blank = grade(p, 0, "start" + " " * 600, tokens=4096, finish="length")
    assert summarize([blank])["cut_whitespace_tail_share"] == 1.0
