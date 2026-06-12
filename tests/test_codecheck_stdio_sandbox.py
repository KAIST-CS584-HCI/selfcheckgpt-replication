from codecheck.execution.stdio_sandbox import run_stdio_batch_in_subprocess, normalize_stdout

ADD = "print(int(input()) + 1)\n"
BOOM = "raise ValueError('boom')\n"
HANG = "while True:\n    pass\n"
ECHO = "import sys\nsys.stdout.write(sys.stdin.read())\n"


def test_runs_each_stdin_and_preserves_order():
    out = run_stdio_batch_in_subprocess(ADD, "", ["1\n", "2\n", "41\n"], timeout=5.0)
    assert out == [("ok", "2"), ("ok", "3"), ("ok", "42")]


def test_empty_stdin_list():
    assert run_stdio_batch_in_subprocess(ADD, "", [], timeout=5.0) == []


def test_nonzero_exit_becomes_error():
    status, _ = run_stdio_batch_in_subprocess(BOOM, "", ["x\n"], timeout=5.0)[0]
    assert status == "err"


def test_infinite_loop_times_out():
    status, _ = run_stdio_batch_in_subprocess(HANG, "", ["x\n"], timeout=1.0)[0]
    assert status == "timeout"


def test_one_timeout_does_not_block_later_inputs():
    out = run_stdio_batch_in_subprocess(
        "import sys\nif sys.stdin.read().strip() == 'hang':\n"
        "    while True: pass\nprint('ok')\n",
        "", ["hang\n", "go\n"], timeout=1.0,
    )
    assert out[0][0] == "timeout"
    assert out[1] == ("ok", "ok")


def test_stdout_passthrough_is_normalized():
    # echo back stdin with trailing whitespace/CRLF -> normalized
    out = run_stdio_batch_in_subprocess(ECHO, "", ["a  \r\nb\n\n"], timeout=5.0)
    assert out == [("ok", "a\nb")]


def test_normalize_stdout_strips_trailing_ws_and_unifies_newlines():
    assert normalize_stdout("1 \r\n2\t\n\n") == "1\n2"
    assert normalize_stdout("\n\nhello\n\n") == "hello"
    assert normalize_stdout("") == ""
