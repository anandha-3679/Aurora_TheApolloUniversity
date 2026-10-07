import pandas as pd
from src.tumor_model import GompertzTumorModel


def test_gompertz_step_growth():
    model = GompertzTumorModel(growth_rate=0.04, carrying_capacity=100.0)
    initial_volume = 15.0
    grown_volume = model.step_growth(initial_volume)
    assert grown_volume > initial_volume
    assert grown_volume < model.carrying_capacity


def test_simulate_trajectory_shape_and_kill():
    model = GompertzTumorModel(default_kill_fraction=0.35)
    traj = model.simulate_trajectory(initial_volume=20.0, horizon_days=28, dose_day=0)
    assert len(traj) == 28
    # On day 0 with dose administered, volume drops immediately by kill fraction
    assert traj[0] < 20.0


def test_cost_of_delay_tradeoff_summary():
    model = GompertzTumorModel()
    initial_volume = 15.0
    delays = [0, 3, 7, 14]

    df = model.simulate_cost_of_delay(
        initial_volume=initial_volume, candidate_delays=delays, horizon_days=28
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df["delay_days"].unique()) == set(delays)

    summary = model.compute_summary_tradeoff(
        initial_volume=initial_volume, candidate_delays=delays
    )
    assert summary[0]["excess_volume_cm3"] == 0.0
    assert summary[0]["regrowth_penalty_pct"] == 0.0

    # Longer delay leads to strictly greater tumor regrowth before treatment
    assert summary[3]["excess_volume_cm3"] > 0
    assert summary[7]["excess_volume_cm3"] > summary[3]["excess_volume_cm3"]
    assert summary[14]["excess_volume_cm3"] > summary[7]["excess_volume_cm3"]
    assert summary[14]["regrowth_penalty_pct"] > summary[7]["regrowth_penalty_pct"]
