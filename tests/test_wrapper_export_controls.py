from __future__ import annotations

from utils.wrapper_citylearn import Wrapper_CityLearn


class _DummyExportEnv:
    def __init__(self) -> None:
        self.render_enabled = True
        self.export_kpis_on_episode_end = True
        self._final_kpis_exported = False
        self.export_calls = []

    def export_final_kpis(self, **kwargs):
        self.export_calls.append(kwargs)
        self._final_kpis_exported = True


def test_wrapper_exports_only_final_episode_with_manual_bau_flags():
    env = _DummyExportEnv()
    wrapper = Wrapper_CityLearn.__new__(Wrapper_CityLearn)
    wrapper.env = env
    wrapper._configured_render_enabled = True
    wrapper._configured_export_kpis_on_episode_end = True
    wrapper._export_final_episode_only = True
    wrapper._export_include_business_as_usual = False
    wrapper._export_business_as_usual_timeseries = False
    wrapper._export_kpi_round_decimals = None
    wrapper._manual_kpi_export = True

    assert wrapper._configure_episode_exports(0, 2) is False
    assert env.render_enabled is False
    assert env.export_kpis_on_episode_end is False

    assert wrapper._configure_episode_exports(1, 2) is True
    assert env.render_enabled is True
    assert env.export_kpis_on_episode_end is False

    wrapper._export_episode_kpis_if_needed(True)

    assert env.export_calls == [
        {
            "include_business_as_usual": False,
            "export_business_as_usual_timeseries": False,
            "kpi_round_decimals": None,
        }
    ]
