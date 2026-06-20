from pathlib import Path

from scripts.analyze_ev_departure_events import _event_rows


def test_event_rows_keeps_repeated_departures_for_same_ev(tmp_path: Path):
    path = tmp_path / "exported_data_building_1_charger_1_1_ep2.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp,EV SOC-%,EV Required SOC Departure-%,EV Departure Time,Is EV Connected,EV Name",
                "2024-08-01T08:00:00,0.80,0.85,0,True,Electric_Vehicle_1",
                "2024-08-02T08:00:00,0.86,0.85,0,True,Electric_Vehicle_1",
                "2024-08-02T09:00:00,0.86,0.85,-1,False,Electric_Vehicle_1",
            ]
        ),
        encoding="utf-8",
    )

    rows = _event_rows(path, tolerance=0.05)

    assert len(rows) == 2
    assert rows[0]["timestamp"] == "2024-08-01T08:00:00"
    assert rows[1]["timestamp"] == "2024-08-02T08:00:00"
    assert rows[0]["success_min_acceptable"] is True
    assert rows[1]["success_min_acceptable"] is True
