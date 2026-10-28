from grits_resolver import cli


def test_cli_forwards_explicit_multitrack_selection(monkeypatch, capsys):
    calls = []

    class Resolver:
        def __init__(self, *roots):
            pass

        def time_join(self, tile, **selection):
            calls.append((tile, selection))
            return []

        def entries(self, sensor, tile):
            return []

    monkeypatch.setattr(cli, "MultiSensorResolver", Resolver)
    cli.main(["--s1-root", "s1", "--s2-root", "s2", "--tile", "49QHE",
              "--s1-track", "18", "--s1-store", "chosen-s1.zarr",
              "--s2-store", "chosen-s2.zarr"])
    tile, selection = calls.pop()
    assert tile == "49QHE"
    assert selection["s1_track"] == "18"
    assert selection["s1_store"] == "chosen-s1.zarr"
    assert selection["s2_store"] == "chosen-s2.zarr"
    assert "Temporal matches: 0/0" in capsys.readouterr().out
