"""Terminal presentation of a run: formatting, plain output when piped, colour rules."""

import pytest

from lunarops import console


@pytest.fixture(autouse=True)
def _plain_environment(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", console.PROGRESS_INTERVAL_VARIABLE, *console._MPI_LAUNCHER_VARIABLES):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0.04, "0.0 s"), (42.12, "42.1 s"), (125, "2 min 05 s"), (3725, "1 h 02 min")],
)
def test_format_duration(seconds, text):
    assert console.format_duration(seconds) == text


def test_format_value_and_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert console.format_value(23650) == "23,650"
    assert console.format_value(True) == "yes"
    assert console.format_value(1.023456) == "1.023"
    assert console.format_path(tmp_path / "out" / "a.txt") == "out/a.txt"
    assert console.format_path("/elsewhere/b.txt") == "/elsewhere/b.txt"


def test_summary_aligns_labels_and_lists_outputs(capsys):
    console.summary({"samples": 23650, "C04 overlaps": 12}, outputs=["/x/a.txt", None, "/x/b.txt"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "    samples       23,650",
        "    C04 overlaps  12",
        "    output        /x/a.txt",
        "                  /x/b.txt",
    ]


def test_piped_output_has_no_ansi_codes(capsys):
    console.program_header(1, 2, "Example")
    console.program_footer(0.5)
    console.run_footer(2, 1.0)
    assert "\033[" not in capsys.readouterr().out


def test_colour_rules(monkeypatch):
    monkeypatch.setattr(console, "_is_tty", lambda stream: True)
    assert console._use_color()
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert not console._use_color()
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert console._use_color()
    monkeypatch.setenv("NO_COLOR", "1")
    assert not console._use_color()


def test_table_prints_header_once_and_dashes_for_missing(capsys):
    table = console.Table((console.Column("iter", 4), console.Column("WRMS [mm]", 9, format=".3f")))
    table.row(1, 18.2341)
    table.row(2, None)
    assert capsys.readouterr().out.splitlines() == [
        "    iter  WRMS [mm]",
        "       1     18.234",
        "       2          -",
    ]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_log_progress_prints_complete_lines_at_the_interval_and_on_completion(capsys):
    clock = _Clock()
    with console.Progress(total=100, desc="O-C", unit="tasks", interactive=False, interval=30, clock=clock) as bar:
        for _ in range(40):
            clock.now += 1.0
            bar.update()
    out = capsys.readouterr().out
    assert "\r" not in out
    assert out.splitlines() == [
        "    O-C  30/100 tasks (30%)  1/s, elapsed 30.0 s, ETA 1 min 10 s",
        "    O-C  40/100 tasks (40%)  done in 40.0 s, 1/s",
    ]


def test_log_progress_reports_nothing_more_when_the_work_fails(capsys):
    def items():
        yield 1
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError):
        list(console.Progress(items(), total=2, desc="x", interactive=False, interval=30))
    assert capsys.readouterr().out == ""


def test_interactive_progress_redraws_in_place_and_clears_around_messages(capsys):
    clock = _Clock()
    bar = console.Progress(total=4, desc="LLR", unit="epochs", interactive=True, clock=clock)
    clock.now = 1.0
    bar.update(2)
    console.info("message")
    bar.close()
    out = capsys.readouterr().out
    assert "\n" not in out.replace("  · message\n", "")
    assert " 50% " in out and "2/4 epochs" in out
    assert out.endswith("\r\033[K")
    assert console._ACTIVE_PROGRESS is None


def test_disabled_progress_counts_silently(capsys):
    assert list(console.Progress(range(3), desc="x", disable=True, interactive=True)) == [0, 1, 2]
    assert capsys.readouterr().out == ""


def test_progress_mode_and_interval_from_environment(monkeypatch):
    monkeypatch.setattr(console, "_is_tty", lambda stream: True)
    assert console.Progress(desc="x").interactive
    monkeypatch.setenv("PMIX_RANK", "0")
    monkeypatch.setenv(console.PROGRESS_INTERVAL_VARIABLE, "5")
    bar = console.Progress(desc="x")
    assert not bar.interactive and bar.interval == 5
    monkeypatch.setenv(console.PROGRESS_INTERVAL_VARIABLE, "soon")
    assert console.Progress(desc="x").interval == 30


def test_failed_run_footer_names_the_program(capsys):
    console.run_footer(3, 12.0, failed="[4/7] LlrResiduals")
    out = capsys.readouterr().out
    assert "run stopped in [4/7] LlrResiduals after 12.0 s (3 program calls completed)" in out
