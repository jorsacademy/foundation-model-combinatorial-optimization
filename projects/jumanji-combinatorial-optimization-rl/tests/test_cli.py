from jumanji_co.cli import main


def test_cli_runs(capsys):
    main()
    output = capsys.readouterr().out
    assert "selected_items=" in output
    assert "total_reward=" in output
    assert "total_weight=" in output
    assert "remaining_budget=" in output
