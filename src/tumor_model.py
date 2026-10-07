from typing import Dict, List
import numpy as np
import pandas as pd


class GompertzTumorModel:
    """Gompertz tumor growth and treatment response simulation model.

    Standard Gompertz differential equation:
    dV/dt = alpha * V * ln(V_max / V)

    Parameters
    ----------
    growth_rate : float
        Intrinsic growth constant alpha (default: 0.04 / day).
    carrying_capacity : float
        Maximum theoretical volume V_max in cm3 (default: 100.0 cm3).
    default_kill_fraction : float
        Fraction of tumor eliminated by the chemotherapy dose (default: 0.35).
    """

    def __init__(
        self,
        growth_rate: float = 0.04,
        carrying_capacity: float = 100.0,
        default_kill_fraction: float = 0.35,
    ):
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.default_kill_fraction = default_kill_fraction

    def step_growth(self, volume: float) -> float:
        """Calculate one day of uninhibited Gompertz growth."""
        v = max(0.1, float(volume))
        growth = self.growth_rate * v * np.log(
            max(1.001, self.carrying_capacity / v)
        )
        return min(self.carrying_capacity, max(0.1, v + growth))

    def simulate_trajectory(
        self,
        initial_volume: float,
        horizon_days: int = 28,
        dose_day: int = 0,
        kill_fraction: float = None,
    ) -> List[float]:
        """Simulate tumor trajectory over `horizon_days` with dose given on `dose_day`.

        During delay period [0, dose_day), tumor regrows uninhibited.
        On `dose_day`, the chemotherapy dose is administered, killing a fraction
        of the current tumor volume.
        """
        if kill_fraction is None:
            kill_fraction = self.default_kill_fraction

        v = max(0.1, float(initial_volume))
        trajectory = []

        for day in range(horizon_days):
            if day == dose_day:
                v = max(0.1, v * (1.0 - kill_fraction))
            else:
                v = self.step_growth(v)
            trajectory.append(round(v, 3))

        return trajectory

    def simulate_cost_of_delay(
        self,
        initial_volume: float,
        candidate_delays: List[int] = None,
        horizon_days: int = 28,
        kill_fraction: float = None,
    ) -> pd.DataFrame:
        """Simulate tumor trajectories across candidate delay options (0/3/7/14 days).

        Returns DataFrame with columns: delay_days, day_index, tumor_volume_cm3.
        """
        if candidate_delays is None:
            candidate_delays = [0, 3, 7, 14]

        records = []
        for delay in candidate_delays:
            traj = self.simulate_trajectory(
                initial_volume=initial_volume,
                horizon_days=horizon_days,
                dose_day=delay,
                kill_fraction=kill_fraction,
            )
            for day_idx, vol in enumerate(traj):
                records.append(
                    {
                        "delay_days": delay,
                        "day_index": day_idx,
                        "tumor_volume_cm3": vol,
                    }
                )

        return pd.DataFrame(records)

    def compute_summary_tradeoff(
        self,
        initial_volume: float,
        candidate_delays: List[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Compute the cost of delay (regrowth before dose administration).

        Evaluates how much the tumor regrows during each delay window relative
        to dosing promptly on day 0.
        """
        if candidate_delays is None:
            candidate_delays = [0, 3, 7, 14]

        v0 = max(0.1, float(initial_volume))
        summary = {}

        for delay in candidate_delays:
            v_curr = v0
            for _ in range(delay):
                v_curr = self.step_growth(v_curr)

            excess_vol = round(v_curr - v0, 3)
            pct_penalty = round(((v_curr - v0) / v0) * 100, 2)
            summary[delay] = {
                "volume_at_dose_cm3": round(v_curr, 3),
                "excess_volume_cm3": max(0.0, excess_vol),
                "regrowth_penalty_pct": max(0.0, pct_penalty),
            }

        return summary
