from typer.testing import CliRunner

from afriembed.cli import app

runner = CliRunner()


def test_help_lists_all_experiments():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in [
        "smoke",
        "afrimteb",
        "speech-text",
        "speech-bitext",
        "fertility",
        "vision",
        "finetune-text",
        "finetune-audio",
        "vocab-trim",
        "rag",
        "index-audio",
        "index-docs",
        "search",
        "ciral",
        "lid",
        "accent",
        "report",
        "list-langs",
    ]:
        assert cmd in r.output, cmd


def test_list_langs_runs():
    r = runner.invoke(app, ["list-langs"])
    assert r.exit_code == 0 and "hau " in r.output and "Hausa" in r.output


def test_no_command_disables_prompts_by_positional_mistake():
    """Every command must pass the real --no-prompts flag (or False) to _opts, never a bare True."""
    import inspect

    import afriembed.cli as cli

    src = inspect.getsource(cli)
    assert "_opts(model, None, True," not in src
    assert "_opts(model, dim, True," not in src
