from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


class GompertzTumorModel:
    """Gompertz tumor growth and patient-specific parameter estimation engine.

    Standard Gompertz differential equation:
    dV/dt = alpha * V * ln(V_max / V)

    Parameters
    ----------
    growth_rate : float
        Intrinsic growth constant alpha (default: 0.042 / day).
    carrying_capacity : float
        Maximum theoretical volume V_max in cm3 (default: 100.0 cm3).
    default_kill_fraction : float
        Fraction of tumor eliminated by the chemotherapy dose (default: 0.35).
    """

    def __init__(
        self,
        growth_rate: float = 0.042,
        carrying_capacity: float = 100.0,
        default_kill_fraction: float = 0.35,
    ):
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.default_kill_fraction = default_kill_fraction

    def step_growth(
        self, volume: float, growth_rate: Optional[float] = None
    ) -> float:
        """Calculate one day of uninhibited Gompertz growth."""
        alpha = self.growth_rate if growth_rate is None else growth_rate
        v = max(0.1, float(volume))
        growth = alpha * v * np.log(max(1.001, self.carrying_capacity / v))
        return min(self.carrying_capacity, max(0.1, v + growth))

    def fit_patient_parameters(
        self,
        clinical_history: pd.DataFrame,
        decision_day: int,
    ) -> Dict[str, float]:
        """Fit patient-specific Gompertz parameters using ONLY day_index <= D.

        Uses nonlinear least squares bounded inside plausible biological ranges.
        Strictly filters clinical_history so no measurements after decision_day
        are accessed.

        Returns
        -------
        dict with keys:
            'growth_rate': fitted alpha
            'kill_fraction': fitted treatment kill effect
            'residual_std': estimated measurement noise standard error
            'volume_at_d': current tumor volume at day D
        """
        # Strict cutoff: zero rows with day_index > decision_day
        hist = clinical_history[
            clinical_history["day_index"] <= decision_day
        ].sort_values("day_index")

        if len(hist) < 3:
            v_d = (
                float(hist.iloc[-1]["tumor_volume_cm3"])
                if not hist.empty
                else 20.0
            )
            return {
                "growth_rate": self.growth_rate,
                "kill_fraction": self.default_kill_fraction,
                "residual_std": 0.45,
                "volume_at_d": v_d,
            }

        obs_v = hist["tumor_volume_cm3"].values.astype(float)
        days = hist["day_index"].values.astype(int)
        doses = set(hist[hist["dose_mg"] > 0]["day_index"].tolist())
        v0 = float(obs_v[0])
        cap = self.carrying_capacity

        def simulate_curve(alpha: float, kill: float) -> np.ndarray:
            v = v0
            res = [v]
            for d in range(1, len(obs_v)):
                day_idx = days[d]
                if day_idx in doses:
                    v = max(0.1, v * (1.0 - kill))
                else:
                    growth = alpha * v * np.log(max(1.001, cap / v))
                    v = min(cap, max(0.1, v + growth))
                res.append(v)
            return np.array(res)

        def residuals(params: np.ndarray) -> np.ndarray:
            return simulate_curve(params[0], params[1]) - obs_v

        # Biological bounds: alpha in [0.015, 0.080], kill in [0.15, 0.55]
        init_guess = [self.growth_rate, self.default_kill_fraction]
        lower_bounds = [0.015, 0.15]
        upper_bounds = [0.080, 0.55]

        opt = least_squares(
            residuals,
            init_guess,
            bounds=(lower_bounds, upper_bounds),
            ftol=1e-5,
            xtol=1e-5,
        )

        a_fit, k_fit = opt.x
        n = len(obs_v)
        p = 2
        residuals_arr = opt.fun
        sigma2 = np.sum(residuals_arr**2) / max(1, n - p)
        residual_std = float(np.clip(np.sqrt(sigma2), 0.20, 1.50))

        return {
            "growth_rate": float(a_fit),
            "kill_fraction": float(k_fit),
            "residual_std": residual_std,
            "volume_at_d": float(obs_v[-1]),
        }

    def forecast_with_uncertainty(
        self,
        current_volume: float,
        candidate_delays: Optional[List[int]] = None,
        growth_rate: Optional[float] = None,
        kill_fraction: Optional[float] = None,
        residual_std: float = 0.45,
        confidence_level: float = 0.90,
    ) -> Dict[int, Dict[str, Any]]:
        """Forecast tumor volume across candidate delay days with 90% uncertainty band.

        Parameters
        ----------
        current_volume : float
            Tumor volume at decision day D (cm3).
        candidate_delays : list of int
            Delays to evaluate (default: [0, 3, 7, 14]).
        growth_rate : float
            Patient-specific fitted growth rate alpha.
        kill_fraction : float
            Patient-specific treatment kill fraction.
        residual_std : float
            Measurement error standard deviation sigma.
        confidence_level : float
            Confidence level (default 0.90 -> z = 1.645).
        """
        if candidate_delays is None:
            candidate_delays = [0, 3, 7, 14]

        alpha = self.growth_rate if growth_rate is None else growth_rate
        kill = (
            self.default_kill_fraction
            if kill_fraction is None
            else kill_fraction
        )

        # 90% two-sided normal quantile: z = 1.645
        z_multiplier = 1.645 if abs(confidence_level - 0.90) < 1e-4 else 1.960
        v0 = max(0.1, float(current_volume))
        forecasts = {}

        for delay in candidate_delays:
            # Simulate uninhibited regrowth during delay days before dose
            v_dose = v0
            for _ in range(delay):
                v_dose = self.step_growth(v_dose, growth_rate=alpha)

            # Measurement & accumulation uncertainty: sigma * sqrt(delay + 1)
            uncertainty_margin = (
                z_multiplier * residual_std * np.sqrt(delay + 1)
            )

            lower_bound = max(0.1, round(v_dose - uncertainty_margin, 2))
            upper_bound = min(
                self.carrying_capacity,
                round(v_dose + uncertainty_margin, 2),
            )
            excess_vol = round(v_dose - v0, 2)
            pct_penalty = round(((v_dose - v0) / v0) * 100.0, 1)

            # Post-treatment nadir volume if dose given at this delay
            v_post = round(max(0.1, v_dose * (1.0 - kill)), 2)

            forecasts[delay] = {
                "delay_days": delay,
                "volume_at_dose_cm3": round(v_dose, 2),
                "lower_bound_90_cm3": lower_bound,
                "upper_bound_90_cm3": upper_bound,
                "excess_volume_cm3": max(0.0, excess_vol),
                "regrowth_penalty_pct": max(0.0, pct_penalty),
                "post_treatment_volume_cm3": v_post,
            }

        return forecasts

    def compute_summary_tradeoff(
        self,
        initial_volume: float,
        candidate_delays: Optional[List[int]] = None,
        growth_rate: Optional[float] = None,
        kill_fraction: Optional[float] = None,
        residual_std: float = 0.45,
    ) -> Dict[int, Dict[str, Any]]:
        """Backwards-compatible wrapper returning forecasts with uncertainty bands."""
        return self.forecast_with_uncertainty(
            current_volume=initial_volume,
            candidate_delays=candidate_delays,
            growth_rate=growth_rate,
            kill_fraction=kill_fraction,
            residual_std=residual_std,
            confidence_level=0.90,
        )

    def evaluate_forecast_mape(
        self,
        clinical_df: pd.DataFrame,
        decision_day: int = 21,
        test_patient_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate forecast error (MAPE) on held-out measurements after day D.

        Compares patient-specific Gompertz fits against a fixed
        population-average Gompertz curve, split across patients.
        """
        all_pids = clinical_df["patient_id"].unique()
        if test_patient_ids is None:
            # Deterministic split for reproducibility if none provided
            np.random.seed(42)
            test_pids = list(
                np.random.choice(
                    all_pids, size=max(5, int(0.25 * len(all_pids))), replace=False
                )
            )
        else:
            test_pids = list(test_patient_ids)

        patient_mapes = []
        population_mapes = []

        pop_alpha = self.growth_rate
        pop_kill = self.default_kill_fraction

        for pid in test_pids:
            p_data = clinical_df[clinical_df["patient_id"] == pid].sort_values(
                "day_index"
            )
            future = p_data[p_data["day_index"] > decision_day]
            if future.empty:
                continue

            # Fit patient-specific model strictly on day_index <= decision_day
            fit = self.fit_patient_parameters(p_data, decision_day=decision_day)
            v_curr_pt = fit["volume_at_d"]
            v_curr_pop = float(
                p_data[p_data["day_index"] <= decision_day].iloc[-1][
                    "tumor_volume_cm3"
                ]
            )

            actuals = future["tumor_volume_cm3"].values.astype(float)
            fut_days = future["day_index"].values.astype(int)
            fut_doses = set(
                future[future["dose_mg"] > 0]["day_index"].tolist()
            )

            pt_preds = []
            pop_preds = []

            for d in fut_days:
                # Patient trajectory
                if d in fut_doses:
                    v_curr_pt = max(0.1, v_curr_pt * (1.0 - fit["kill_fraction"]))
                else:
                    v_curr_pt = self.step_growth(
                        v_curr_pt, growth_rate=fit["growth_rate"]
                    )
                pt_preds.append(v_curr_pt)

                # Population trajectory
                if d in fut_doses:
                    v_curr_pop = max(0.1, v_curr_pop * (1.0 - pop_kill))
                else:
                    v_curr_pop = self.step_growth(
                        v_curr_pop, growth_rate=pop_alpha
                    )
                pop_preds.append(v_curr_pop)

            pt_err = (
                np.mean(np.abs(np.array(pt_preds) - actuals) / actuals) * 100.0
            )
            pop_err = (
                np.mean(np.abs(np.array(pop_preds) - actuals) / actuals) * 100.0
            )

            patient_mapes.append(pt_err)
            population_mapes.append(pop_err)

        return {
            "patient_specific_mape_mean": float(np.mean(patient_mapes)),
            "patient_specific_mape_std": float(np.std(patient_mapes)),
            "population_average_mape_mean": float(np.mean(population_mapes)),
            "population_average_mape_std": float(np.std(population_mapes)),
            "num_test_patients": len(patient_mapes),
        }
