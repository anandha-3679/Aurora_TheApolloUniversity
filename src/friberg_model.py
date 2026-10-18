from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


from scipy.integrate import solve_ivp

# Accurate ODE settings used wherever gradients are taken (fitting) or a single
# primary trajectory is produced. Loose tolerances made finite-difference
# Jacobians pure noise, which froze the optimizer at its initial guess.
ODE_KW = {"method": "LSODA", "rtol": 1e-7, "atol": 1e-9, "max_step": 0.5}
# Cheaper settings for Monte Carlo sampling (no gradients needed).
ODE_KW_MC = {"method": "LSODA", "rtol": 1e-4, "atol": 1e-6, "max_step": 0.5}


def drug_effect(
    t: float, slope: float, dose_schedule: List[tuple], k_decay: float
) -> float:
    """Saturating (Emax) drug effect in [0, 1).

    C(t) = sum dose/100 * exp(-k_decay * (t - t_dose)); E = slope*C / (1 + slope*C).
    Unlike the old linear E = slope*C, this can never exceed 100% kill.
    """
    c_drug = 0.0
    for d_time, d_amount in dose_schedule:
        if t >= d_time:
            c_drug += (d_amount / 100.0) * np.exp(-k_decay * (t - d_time))
    sc = max(0.0, slope) * c_drug
    return sc / (1.0 + sc)


def friberg_compartmental_rhs(
    t: float,
    y: List[float],
    base_anc: float,
    ktr: float,
    gamma: float,
    slope: float,
    dose_schedule: List[tuple],
    k_decay: float = 0.7,
) -> List[float]:
    """Right-hand side of Friberg myelosuppression compartmental ODE system.

    Parameters:
    Prol: proliferating stem/progenitor cells
    T1, T2, T3: maturation transit stages
    Circ: circulating mature neutrophils (ANC)

    Equations:
    ktr = 4 / MTT
    dProl/dt = ktr * Prol * (1 - E_drug) * (Base/Circ)^gamma - ktr * Prol
    dT1/dt   = ktr * (Prol - T1)
    dT2/dt   = ktr * (T1 - T2)
    dT3/dt   = ktr * (T2 - T3)
    dCirc/dt = ktr * (T3 - Circ)
    E_drug   = slope*C / (1 + slope*C)   (Emax, bounded below 1)
    """
    Prol, T1, T2, T3, Circ = y

    e_drug = drug_effect(t, slope, dose_schedule, k_decay)
    circ_safe = max(1e-4, Circ)
    fb = (base_anc / circ_safe) ** gamma

    d_prol = ktr * Prol * (1.0 - e_drug) * fb - ktr * Prol
    d_t1 = ktr * (Prol - T1)
    d_t2 = ktr * (T1 - T2)
    d_t3 = ktr * (T2 - T3)
    d_circ = ktr * (T3 - Circ)

    return [d_prol, d_t1, d_t2, d_t3, d_circ]


class FribergANCModel:
    """Semi-mechanistic Friberg-style myelosuppression compartmental model.

    Models post-chemotherapy neutrophil kinetics using ODE compartmental stages:
    - Base: baseline circulating neutrophils (Base)
    - MTT: mean transit time through maturation compartments (ktr = 4 / MTT)
    - gamma: feedback exponent governing proliferation feedback (Base / Circ)^gamma
    - slope: drug concentration myelosuppressive kill potency

    Equations:
    ktr = 4 / MTT
    dProl/dt  = ktr * Prol * (1 - E_drug) * (Base/Circ)^gamma - ktr * Prol
    dT1/dt    = ktr * (Prol - T1)
    dT2/dt    = ktr * (T1 - T2)
    dT3/dt    = ktr * (T2 - T3)
    dCirc/dt  = ktr * (T3 - Circ)      # Circ = ANC
    E_drug = slope*C/(1+slope*C), C a decaying pulse (half-life ~1 day).

    Defaults follow published Friberg estimates (MTT ~ 5 d, gamma ~ 0.17).
    """

    # Fit bounds: MTT (days), gamma, slope. In literature gamma is tightly between 0.12-0.24.
    LOWER_BOUNDS = (4.0, 0.10, 0.5)
    UPPER_BOUNDS = (14.0, 0.35, 10.0)

    def __init__(
        self,
        default_base_anc: float = 4.5,
        default_mtt: float = 5.0,
        default_recovery_rate: float = 0.17,
        default_drug_slope: float = 4.0,
        k_decay: float = 0.7,
    ):
        self.default_base_anc = default_base_anc
        self.default_mtt = default_mtt
        self.default_recovery_rate = default_recovery_rate
        self.default_gamma = default_recovery_rate
        self.default_drug_slope = default_drug_slope
        self.k_decay = k_decay

    @staticmethod
    def simulate_anc_trajectory(
        time_days: np.ndarray,
        base_anc: float,
        mtt: float,
        recovery_rate: Optional[float] = None,
        drug_effect: Optional[float] = None,
        gamma: Optional[float] = None,
        slope: Optional[float] = None,
        dose_mg: float = 100.0,
        k_decay: float = 0.7,
    ) -> np.ndarray:
        """Simulate post-dose ANC trajectory across time_days using solve_ivp.

        Parameters
        ----------
        time_days : np.ndarray
            Days relative to dose (e.g. 0, 1, 2, ...).
        base_anc : float
            Circulating baseline ANC (Base).
        mtt : float
            Mean transit time (MTT).
        recovery_rate : Optional[float]
            Hematopoietic recovery rate parameter / feedback exponent gamma.
        drug_effect : Optional[float]
            Suppression slope / drug kill factor.
        gamma : Optional[float]
            Explicit feedback exponent gamma (Base/Circ)^gamma.
        slope : Optional[float]
            Explicit drug slope.
        dose_mg : float
            Administered pulse dose amount.
        k_decay : float
            Elimination rate for the decaying drug pulse.
        """
        t_arr = np.maximum(0.0, np.asarray(time_days, dtype=float))
        if len(t_arr) == 0:
            return np.array([], dtype=float)

        b_val = max(0.1, float(base_anc))
        mtt_val = max(1.0, float(mtt))
        ktr_val = 4.0 / mtt_val

        g_val = (
            float(gamma)
            if gamma is not None
            else (float(recovery_rate) if recovery_rate is not None else 0.17)
        )
        s_val = (
            float(slope)
            if slope is not None
            else (float(drug_effect) if drug_effect is not None else 4.0)
        )

        # If zero dose or no drug effect, steady-state ANC = Base
        if s_val <= 0.0 or dose_mg <= 0.0:
            return np.full_like(t_arr, fill_value=b_val)

        dose_schedule = [(0.0, float(dose_mg))]

        def rhs(t, y):
            return friberg_compartmental_rhs(
                t=t,
                y=y,
                base_anc=b_val,
                ktr=ktr_val,
                gamma=g_val,
                slope=s_val,
                dose_schedule=dose_schedule,
                k_decay=k_decay,
            )

        y0 = [b_val, b_val, b_val, b_val, b_val]
        sort_idx = np.argsort(t_arr)
        t_sorted = t_arr[sort_idx]
        unique_t, inv_map = np.unique(t_sorted, return_inverse=True)

        t_min = float(min(0.0, unique_t[0]))
        t_max = float(max(0.01, unique_t[-1]))

        sol = solve_ivp(
            rhs,
            (t_min, t_max),
            y0,
            t_eval=unique_t,
            **ODE_KW,
        )

        if sol.success:
            circ_unique = sol.y[4]
            circ_sorted = circ_unique[inv_map]
            res = np.empty_like(t_arr)
            res[sort_idx] = circ_sorted
            return np.maximum(0.1, res)
        return np.full_like(t_arr, fill_value=b_val)

    def simulate_longitudinal_anc(
        self,
        days: np.ndarray,
        dose_days: List[int],
        dose_amounts: List[float],
        base_anc: float,
        mtt: float,
        recovery_rate: Optional[float] = None,
        drug_slope: Optional[float] = None,
        gamma: Optional[float] = None,
        slope: Optional[float] = None,
    ) -> np.ndarray:
        """Simulate longitudinal ANC over multi-dose clinical time series via solve_ivp."""
        days_arr = np.asarray(days, dtype=float)
        if len(days_arr) == 0:
            return np.array([], dtype=float)

        b_val = max(0.1, float(base_anc))
        mtt_val = max(1.0, float(mtt))
        ktr_val = 4.0 / mtt_val

        g_val = (
            float(gamma)
            if gamma is not None
            else (
                float(recovery_rate)
                if recovery_rate is not None
                else self.default_recovery_rate
            )
        )
        s_val = (
            float(slope)
            if slope is not None
            else (
                float(drug_slope) if drug_slope is not None else self.default_drug_slope
            )
        )

        valid_doses = [
            (float(td), float(dm)) for td, dm in zip(dose_days, dose_amounts) if dm > 0
        ]

        if len(valid_doses) == 0 or s_val <= 0.0:
            return np.full_like(days_arr, fill_value=b_val)

        def rhs(t, y):
            return friberg_compartmental_rhs(
                t=t,
                y=y,
                base_anc=b_val,
                ktr=ktr_val,
                gamma=g_val,
                slope=s_val,
                dose_schedule=valid_doses,
                k_decay=self.k_decay,
            )

        y0 = [b_val, b_val, b_val, b_val, b_val]
        sort_idx = np.argsort(days_arr)
        t_sorted = days_arr[sort_idx]
        unique_t, inv_map = np.unique(t_sorted, return_inverse=True)

        t_min = float(min(0.0, unique_t[0]))
        t_max = float(max(0.01, unique_t[-1]))

        sol = solve_ivp(
            rhs,
            (t_min, t_max),
            y0,
            t_eval=unique_t,
            **ODE_KW,
        )

        if sol.success:
            circ_unique = sol.y[4]
            circ_sorted = circ_unique[inv_map]
            res = np.empty_like(days_arr)
            res[sort_idx] = circ_sorted
            return np.maximum(0.1, res)
        return np.full_like(days_arr, fill_value=b_val)

    @staticmethod
    def compute_marrow_strain(
        wearable_history: pd.DataFrame,
        decision_day: Optional[int] = None,
    ) -> float:
        """Compute the marrow strain index from longitudinal autonomic wearable trends.

        Under chemotherapy-induced bone-marrow suppression:
        - Autonomic strain manifests as elevated resting heart rate (RHR ratio > 1.0),
        - Suppressed parasympathetic tone (HRV ratio < 1.0),
        - Reduced mobility/fatigue (steps ratio < 1.0).

        Strain index:
        S = (
            0.45 * max(0, rhr_ratio - 1.0)
            + 0.35 * max(0, 1.0 - hrv_ratio)
            + 0.20 * max(0, 1.0 - steps_ratio)
        )
        where ratios represent 7-day trailing mean vs 30-day baseline (<= D).

        Returns
        -------
        float in [0.0, 1.0], where higher values signify severe marrow strain.
        """
        if wearable_history is None or wearable_history.empty:
            return 0.0

        w_df = wearable_history.copy()
        if decision_day is not None and "day_index" in w_df.columns:
            w_df = w_df[w_df["day_index"] <= decision_day]

        if w_df.empty:
            return 0.0

        w_df = w_df.sort_values("day_index")

        # Baseline: first 30 available days (or overall mean if < 30 days)
        # Recent state: trailing 7 days up to decision_day
        b_df = w_df.head(30) if len(w_df) >= 14 else w_df
        r_df = w_df.tail(7)

        b30_rhr = (
            float(b_df["resting_heart_rate"].mean())
            if "resting_heart_rate" in b_df.columns
            else 70.0
        )
        b30_hrv = (
            float(b_df["hrv_rmssd"].mean()) if "hrv_rmssd" in b_df.columns else 45.0
        )
        b30_steps = (
            float(b_df["step_count"].mean()) if "step_count" in b_df.columns else 6000.0
        )

        m7_rhr = (
            float(r_df["resting_heart_rate"].mean())
            if "resting_heart_rate" in r_df.columns
            else b30_rhr
        )
        m7_hrv = (
            float(r_df["hrv_rmssd"].mean()) if "hrv_rmssd" in r_df.columns else b30_hrv
        )
        m7_steps = (
            float(r_df["step_count"].mean())
            if "step_count" in r_df.columns
            else b30_steps
        )

        rhr_ratio = m7_rhr / b30_rhr if b30_rhr > 0 else 1.0
        hrv_ratio = m7_hrv / b30_hrv if b30_hrv > 0 else 1.0
        steps_ratio = m7_steps / b30_steps if b30_steps > 0 else 1.0

        strain = (
            0.45 * max(0.0, rhr_ratio - 1.0)
            + 0.35 * max(0.0, 1.0 - hrv_ratio)
            + 0.20 * max(0.0, 1.0 - steps_ratio)
        )
        return float(np.clip(strain, 0.0, 1.0))

    def fit_patient_anc(
        self,
        patient_history: pd.DataFrame,
        decision_day: Optional[int] = None,
        baseline_anc_prior: Optional[float] = None,
        wearable_history: Optional[pd.DataFrame] = None,
        strain_nudge_weight: float = 0.30,
    ) -> Dict[str, float]:
        """Fit patient-specific Friberg parameters from observed history.

        Strictly filters clinical and wearable history so no measurements after
        decision_day are accessed when decision_day is provided.

        Parameters
        ----------
        patient_history : pd.DataFrame
            Must contain 'day_index', 'anc', and 'dose_mg'.
        decision_day : Optional[int]
            Cutoff day D. If specified, only day_index <= decision_day are used.
        baseline_anc_prior : Optional[float]
            Known patient profile baseline ANC prior, if available.
        wearable_history : Optional[pd.DataFrame]
            Longitudinal wearable records (day_index, resting_heart_rate, etc.).
        strain_nudge_weight : float
            Maximum fractional downward nudge applied to the recovery rate under
            elevated marrow strain (default 0.30, i.e., up to -30% adjustment).

        Returns
        -------
        dict with fitted parameters:
            'base_anc': fitted baseline ANC (x10^9 / L)
            'mtt': fitted transit time (days, nadir delay)
            'recovery_rate': wearable-nudged effective recovery rate
            'raw_recovery_rate': un-nudged recovery rate from ANC fits alone
            'marrow_strain': physiological marrow strain index in [0, 1]
            'drug_slope': fitted drug myelosuppression slope
            'residual_std': estimated measurement residual std error
        """
        hist = patient_history.copy()
        if decision_day is not None:
            hist = hist[hist["day_index"] <= decision_day]
        hist = hist.dropna(subset=["day_index", "anc"]).sort_values("day_index")

        default_base = (
            float(baseline_anc_prior)
            if baseline_anc_prior is not None
            else self.default_base_anc
        )

        marrow_strain = (
            self.compute_marrow_strain(wearable_history, decision_day)
            if wearable_history is not None
            else 0.0
        )

        # Fallback if insufficient historical observations
        default_cov = np.diag([0.2**2, 1.0**2, 0.05**2, 1.5**2])
        if len(hist) < 3:
            raw_rec = self.default_recovery_rate
            nudged_rec = float(
                np.clip(
                    raw_rec * (1.0 - strain_nudge_weight * marrow_strain),
                    0.05,
                    2.0,
                )
            )
            return {
                "base_anc": default_base,
                "Base": default_base,
                "mtt": self.default_mtt,
                "MTT": self.default_mtt,
                "gamma": float(np.round(nudged_rec, 3)),
                "raw_gamma": float(np.round(raw_rec, 3)),
                "recovery_rate": float(np.round(nudged_rec, 3)),
                "raw_recovery_rate": float(np.round(raw_rec, 3)),
                "marrow_strain": float(np.round(marrow_strain, 3)),
                "drug_slope": self.default_drug_slope,
                "slope": self.default_drug_slope,
                "residual_std": 0.15,
                "parameter_covariance": default_cov.tolist(),
                "covariance_matrix": default_cov,
            }

        days = hist["day_index"].values.astype(float)
        obs_anc = hist["anc"].values.astype(float)

        dosing_rows = hist[hist["dose_mg"] > 0]
        dose_days = dosing_rows["day_index"].tolist()
        dose_amounts = dosing_rows["dose_mg"].tolist()

        # If no doses yet administered, fit only baseline ANC
        if len(dose_days) == 0:
            fitted_base = float(np.clip(np.median(obs_anc), 2.0, 7.5))
            resid = float(np.std(obs_anc - fitted_base))
            raw_rec = self.default_recovery_rate
            nudged_rec = float(
                np.clip(
                    raw_rec * (1.0 - strain_nudge_weight * marrow_strain),
                    0.05,
                    2.0,
                )
            )
            return {
                "base_anc": fitted_base,
                "Base": fitted_base,
                "mtt": self.default_mtt,
                "MTT": self.default_mtt,
                "gamma": float(np.round(nudged_rec, 3)),
                "raw_gamma": float(np.round(raw_rec, 3)),
                "recovery_rate": float(np.round(nudged_rec, 3)),
                "raw_recovery_rate": float(np.round(raw_rec, 3)),
                "marrow_strain": float(np.round(marrow_strain, 3)),
                "drug_slope": self.default_drug_slope,
                "slope": self.default_drug_slope,
                "residual_std": max(0.05, resid),
                "parameter_covariance": default_cov.tolist(),
                "covariance_matrix": default_cov,
            }

        # Base is fixed to patient's measured baseline_anc from profile
        fixed_base = default_base

        def model_func(params: np.ndarray) -> np.ndarray:
            mtt_val, rec_val, d_slope = params
            return self.simulate_longitudinal_anc(
                days=days,
                dose_days=dose_days,
                dose_amounts=dose_amounts,
                base_anc=fixed_base,
                mtt=mtt_val,
                recovery_rate=rec_val,
                drug_slope=d_slope,
            )

        # Fit in log-parameter space (all parameters positive, very different
        # scales). Weak log-normal priors keep weakly-informed parameters sane.
        prior_mean_log = np.log(
            [self.default_mtt, self.default_recovery_rate, self.default_drug_slope]
        )
        prior_sd_log = np.array([0.35, 0.25, 0.70])
        prior_weight = np.sqrt(max(1.0, 0.05 * len(obs_anc)))

        def residuals(log_params: np.ndarray) -> np.ndarray:
            data_res = model_func(np.exp(log_params)) - obs_anc
            prior_res = (
                prior_weight * 0.15 * (log_params - prior_mean_log) / prior_sd_log
            )
            return np.concatenate([data_res, prior_res])

        lower_bounds = np.log(self.LOWER_BOUNDS)
        upper_bounds = np.log(self.UPPER_BOUNDS)
        init_guess = [
            self.default_mtt,
            self.default_recovery_rate,
            self.default_drug_slope,
        ]
        starts = [init_guess, [7.0, 0.22, 6.0], [4.5, 0.14, 1.8]]

        try:
            opt = None
            for x0 in starts:
                x0_log = np.clip(np.log(x0), lower_bounds + 1e-6, upper_bounds - 1e-6)
                cand = least_squares(
                    residuals,
                    x0_log,
                    bounds=(lower_bounds, upper_bounds),
                    diff_step=1e-3,
                    x_scale="jac",
                    max_nfev=200,
                )
                if opt is None or cand.cost < opt.cost:
                    opt = cand
            mtt_fit, rec_fit, slope_fit = np.exp(opt.x)
            res_arr = opt.fun[: len(obs_anc)]
            dof = max(1, len(obs_anc) - 3)
            s_sq = float(np.sum(res_arr**2) / dof)
            res_std = float(np.clip(np.sqrt(s_sq), 0.05, 2.0))

            # Parameter covariance: sigma^2 * (J^T J)^(-1) with Bayesian prior regularization.
            # Jacobian is w.r.t. log-params; map to natural scale with diag(theta).
            J = (
                opt.jac[: len(obs_anc)] * np.exp(opt.x)[None, :]
                if opt.jac is not None
                else None
            )
            if J is not None and J.shape[0] >= 3 and J.shape[1] == 3:
                JTJ = J.T @ J
                prior_stds = np.array([1.5, 0.08, 2.5])
                prior_prec = np.diag(1.0 / (prior_stds**2))
                info_mat = (JTJ / max(1e-4, s_sq)) + prior_prec
                cov_mat_3 = np.linalg.pinv(info_mat)
                cov_mat_3 = 0.5 * (cov_mat_3 + cov_mat_3.T)
                eigvals, eigvecs = np.linalg.eigh(cov_mat_3)
                eigvals = np.maximum(eigvals, 1e-6)
                param_cov_3 = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Construct 4x4 matrix with zero variance on fixed Base
                param_cov = np.zeros((4, 4), dtype=float)
                param_cov[1:, 1:] = param_cov_3
            else:
                param_cov = np.diag([0.0, 1.0**2, 0.05**2, 1.5**2])
        except Exception:
            mtt_fit, rec_fit, slope_fit = init_guess
            res_std = 0.15
            param_cov = np.diag([0.0, 1.0**2, 0.05**2, 1.5**2])

        raw_rec = float(rec_fit)
        # Nudge recovery rate downward based on autonomic marrow strain:
        # r_nudged = r_raw * (1 - lambda * marrow_strain)
        nudged_rec = float(
            np.clip(
                raw_rec * (1.0 - strain_nudge_weight * marrow_strain),
                0.05,
                2.0,
            )
        )

        return {
            "base_anc": float(np.round(fixed_base, 3)),
            "Base": float(np.round(fixed_base, 3)),
            "mtt": float(np.round(mtt_fit, 2)),
            "MTT": float(np.round(mtt_fit, 2)),
            "gamma": float(np.round(nudged_rec, 3)),
            "raw_gamma": float(np.round(raw_rec, 3)),
            "recovery_rate": float(np.round(nudged_rec, 3)),
            "raw_recovery_rate": float(np.round(raw_rec, 3)),
            "marrow_strain": float(np.round(marrow_strain, 3)),
            "drug_slope": float(np.round(slope_fit, 3)),
            "slope": float(np.round(slope_fit, 3)),
            "residual_std": float(np.round(res_std, 3)),
            "parameter_covariance": param_cov.tolist(),
            "covariance_matrix": param_cov,
        }

    def get_compartment_state(
        self,
        base_anc: float,
        mtt: float,
        gamma: Optional[float] = None,
        slope: Optional[float] = None,
        patient_history: Optional[pd.DataFrame] = None,
        decision_day: Optional[int] = None,
        recovery_rate: Optional[float] = None,
        drug_slope: Optional[float] = None,
        wearable_history: Optional[pd.DataFrame] = None,
        use_filter: bool = True,
    ) -> List[float]:
        """Integrate through observed history to get the 5-compartment state at day D.

        When `use_filter=True` (default), employs an Extended Kalman Filter (EKF) observer
        that assimilates intermediate ANC laboratory observations, correcting both circulating
        neutrophils and upstream transit/proliferative compartments in real time.

        Returns [Prol, T1, T2, T3, Circ] at t = decision_day.
        """
        b_val = max(0.1, float(base_anc))
        mtt_val = max(1.0, float(mtt))
        ktr_val = 4.0 / mtt_val

        g_val = (
            float(gamma)
            if gamma is not None
            else (
                float(recovery_rate)
                if recovery_rate is not None
                else self.default_recovery_rate
            )
        )
        s_val = (
            float(slope)
            if slope is not None
            else (
                float(drug_slope) if drug_slope is not None else self.default_drug_slope
            )
        )

        d_day = 0.0 if decision_day is None else max(0.0, float(decision_day))

        # Default resting steady state if no history or at day 0
        if (
            patient_history is None
            or (hasattr(patient_history, "empty") and patient_history.empty)
            or d_day <= 0
        ):
            return [b_val, b_val, b_val, b_val, b_val]

        hist = patient_history.copy()
        if "day_index" in hist.columns:
            hist = hist[hist["day_index"] <= d_day].sort_values("day_index")
        if "dose_mg" in hist.columns:
            dosing_rows = hist[hist["dose_mg"] > 0]
            valid_doses = [
                (float(td), float(dm))
                for td, dm in zip(
                    dosing_rows["day_index"].tolist(), dosing_rows["dose_mg"].tolist()
                )
                if dm > 0
            ]
        else:
            valid_doses = []

        if len(valid_doses) == 0 or s_val <= 0.0:
            return [b_val, b_val, b_val, b_val, b_val]

        # -------------------------------------------------------------
        # EKF State Filter: Assimilate Observed Laboratory ANC Draws
        # -------------------------------------------------------------
        obs_rows = (
            hist.dropna(subset=["anc"]).to_dict("records")
            if (use_filter and "anc" in hist.columns)
            else []
        )

        if use_filter and len(obs_rows) > 0:
            x = np.array([b_val] * 5, dtype=float)
            P = np.diag([0.2**2 * b_val**2] * 5)
            t_curr = 0.0

            for row in obs_rows:
                t_next = float(row["day_index"])
                y_obs = float(row["anc"])
                dt = t_next - t_curr

                if dt > 0:
                    sol = solve_ivp(
                        lambda t, y: friberg_compartmental_rhs(
                            t,
                            y,
                            b_val,
                            ktr_val,
                            g_val,
                            s_val,
                            valid_doses,
                            self.k_decay,
                        ),
                        (t_curr, t_next),
                        x,
                        t_eval=[t_next],
                        **ODE_KW,
                    )
                    x_pred = sol.y[:, -1] if sol.success else x

                    # Compute linearized Jacobian F at x_pred
                    P_c, T1_c, T2_c, T3_c, Circ_c = x_pred
                    circ_safe = max(1e-4, Circ_c)
                    e_dr = drug_effect(t_next, s_val, valid_doses, self.k_decay)
                    fb = (b_val / circ_safe) ** g_val

                    F = np.zeros((5, 5))
                    F[0, 0] = ktr_val * ((1.0 - e_dr) * fb - 1.0)
                    F[0, 4] = (
                        -ktr_val
                        * g_val
                        * P_c
                        * (1.0 - e_dr)
                        * (b_val**g_val)
                        / (circ_safe ** (g_val + 1))
                    )
                    F[1, 0] = ktr_val
                    F[1, 1] = -ktr_val
                    F[2, 1] = ktr_val
                    F[2, 2] = -ktr_val
                    F[3, 2] = ktr_val
                    F[3, 3] = -ktr_val
                    F[4, 3] = ktr_val
                    F[4, 4] = -ktr_val

                    Phi = np.eye(5) + F * dt
                    Q = np.diag([0.05 * b_val] * 5) * dt
                    P_pred = Phi @ P @ Phi.T + Q
                    x = x_pred
                    P = P_pred
                    t_curr = t_next

                # Measurement update with y_obs
                H = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
                R = max(0.02, (0.08 * y_obs) ** 2)
                S = float((H @ P @ H.T)[0, 0] + R)
                K = (P @ H.T) / S  # shape (5, 1)

                r = y_obs - x[4]
                x = x + (K.flatten() * r)
                x = np.maximum(0.05, x)

                I_KH = np.eye(5) - K @ H
                P = I_KH @ P @ I_KH.T + (K * R) @ K.T

            # Propagate forward from last observation day to d_day
            if d_day > t_curr:
                sol = solve_ivp(
                    lambda t, y: friberg_compartmental_rhs(
                        t, y, b_val, ktr_val, g_val, s_val, valid_doses, self.k_decay
                    ),
                    (t_curr, d_day),
                    x,
                    t_eval=[d_day],
                    **ODE_KW,
                )
                if sol.success:
                    x = np.maximum(0.05, sol.y[:, -1])

            # Optional autonomic marrow strain fusion from wearables
            if wearable_history is not None and not wearable_history.empty:
                strain = self.compute_marrow_strain(wearable_history, int(d_day))
                if strain > 0.15:
                    x[:4] = x[:4] * (1.0 - 0.20 * strain)
                    x = np.maximum(0.05, x)

            return [float(v) for v in x]

        # Fallback to pure open-loop forward ODE integration if filter not requested
        def rhs(t, y):
            return friberg_compartmental_rhs(
                t=t,
                y=y,
                base_anc=b_val,
                ktr=ktr_val,
                gamma=g_val,
                slope=s_val,
                dose_schedule=valid_doses,
                k_decay=self.k_decay,
            )

        y0 = [b_val, b_val, b_val, b_val, b_val]
        sol = solve_ivp(
            rhs,
            (0.0, d_day),
            y0,
            t_eval=[d_day],
            **ODE_KW,
        )

        if sol.success:
            return [float(sol.y[i][-1]) for i in range(5)]
        return [b_val, b_val, b_val, b_val, b_val]

    def forecast_anc_nadir(
        self,
        base_anc: float,
        mtt: float,
        recovery_rate: Optional[float] = None,
        candidate_delays: Optional[List[int]] = None,
        dose_mg: float = 100.0,
        drug_slope: Optional[float] = None,
        gamma: Optional[float] = None,
        slope: Optional[float] = None,
        horizon_days: int = 21,
        threshold: float = 1.0,
        patient_history: Optional[pd.DataFrame] = None,
        decision_day: Optional[int] = None,
        initial_compartment_state: Optional[List[float]] = None,
        parameter_covariance: Optional[Any] = None,
        num_mc_samples: int = 200,
        random_seed: int = 42,
        wearable_history: Optional[pd.DataFrame] = None,
        use_filter: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """Forecast forward post-dose ANC trajectory across candidate delay scenarios.

        Simulates from the patient's current compartment state at day D.
        Uses parameter covariance matrix sigma^2 * (J^T J)^(-1) from the fit to sample
        `num_mc_samples` (default 200) parameter sets and simulate each one.
        P(severe neutropenia) for each delay is computed as the fraction of simulation
        runs where the nadir drops below 1.0. Also reports a 90% nadir band and expected
        days below 1.0.

        Parameters
        ----------
        base_anc : float
            Baseline circulating neutrophil count (Base).
        mtt : float
            Maturation transit time (MTT).
        recovery_rate : Optional[float]
            Feedback exponent gamma (or effective recovery rate).
        candidate_delays : Optional[List[int]]
            Candidate delay offsets from decision day D (default [0, 3, 7, 14]).
        dose_mg : float
            Administered chemotherapy dose for the upcoming cycle.
        drug_slope : Optional[float]
            Drug-induced myelosuppression slope.
        gamma : Optional[float]
            Explicit feedback exponent gamma.
        slope : Optional[float]
            Explicit drug slope.
        horizon_days : int
            Forward simulation horizon (days post-dose, default 21).
        threshold : float
            ANC threshold defining severe neutropenia (default 1.0 x 10^9 / L).
        patient_history : Optional[pd.DataFrame]
            Observed patient clinical history up to day D.
        decision_day : Optional[int]
            Current decision day D.
        initial_compartment_state : Optional[List[float]]
            Explicit [Prol, T1, T2, T3, Circ] state at day D if precomputed.
        parameter_covariance : Optional[Any]
            Parameter covariance matrix sigma^2 * (J^T J)^(-1) for Base, MTT, gamma, slope.
        num_mc_samples : int
            Number of Monte Carlo parameter sets to sample and simulate (default 200).
        random_seed : int
            Random seed for reproducibility of parameter sampling.

        Returns
        -------
        dict mapping delay -> {
            'curve': np.ndarray of length horizon_days + 1 (median / primary trajectory),
            'time_days': list of days relative to dose,
            'predicted_nadir_anc': float,
            'predicted_nadir_day': int,
            'days_below_1_0': int,
            'days_below_threshold': int,
            'severe_neutropenia_risk': float (simulation-based P(nadir < 1.0)),
            'simulated_severe_risk': float,
            'nadir_band': List[float] (5th and 95th percentiles of nadir),
            'nadir_lower_90': float,
            'nadir_upper_90': float,
            'expected_days_below_1_0': float,
            'pre_dose_anc': float,
            'state_at_decision_day': List[float],
        }
        """
        if candidate_delays is None:
            candidate_delays = [0, 3, 7, 14]

        eff_slope = (
            float(slope)
            if slope is not None
            else (
                float(drug_slope) if drug_slope is not None else self.default_drug_slope
            )
        )
        base_gamma = (
            float(gamma)
            if gamma is not None
            else (
                float(recovery_rate)
                if recovery_rate is not None
                else self.default_recovery_rate
            )
        )

        b_val = max(0.1, float(base_anc))
        mtt_val = max(1.0, float(mtt))
        ktr_val = 4.0 / mtt_val
        d_val = 0.0 if decision_day is None else max(0.0, float(decision_day))

        # 1. Integrate through observed history to get primary compartment state at day D
        if (
            initial_compartment_state is not None
            and len(initial_compartment_state) == 5
        ):
            state_at_D = [float(v) for v in initial_compartment_state]
        else:
            state_at_D = self.get_compartment_state(
                base_anc=b_val,
                mtt=mtt_val,
                gamma=base_gamma,
                slope=eff_slope,
                patient_history=patient_history,
                decision_day=int(d_val),
                wearable_history=wearable_history,
                use_filter=use_filter,
            )

        # 2. Extract valid prior dose history up to day D for Monte Carlo sample state integration
        prior_doses = []
        if patient_history is not None and not (
            hasattr(patient_history, "empty") and patient_history.empty
        ):
            p_hist = patient_history[patient_history["day_index"] <= d_val]
            if "dose_mg" in p_hist.columns:
                d_rows = p_hist[p_hist["dose_mg"] > 0]
                prior_doses = [
                    (float(td), float(dm))
                    for td, dm in zip(
                        d_rows["day_index"].tolist(), d_rows["dose_mg"].tolist()
                    )
                    if dm > 0
                ]

        # 3. Setup Covariance Matrix and Sample Parameter Sets
        if parameter_covariance is not None:
            cov_arr = np.array(parameter_covariance, dtype=float)
            if cov_arr.shape != (4, 4):
                cov_arr = np.diag([0.15**2, 1.0**2, 0.05**2, 1.5**2])
        else:
            cov_arr = np.diag([0.15**2, 1.0**2, 0.05**2, 1.5**2])

        cov_arr = 0.5 * (cov_arr + cov_arr.T)
        eigvals, eigvecs = np.linalg.eigh(cov_arr)
        eigvals = np.maximum(eigvals, 1e-6)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        rng = np.random.default_rng(random_seed)
        mean_params = np.array([b_val, mtt_val, base_gamma, eff_slope], dtype=float)
        mc_samples = rng.multivariate_normal(
            mean_params, cov_psd, size=max(10, num_mc_samples)
        )
        mc_samples[:, 0] = np.clip(mc_samples[:, 0], 2.0, 7.5)
        mc_samples[:, 1] = np.clip(mc_samples[:, 1], 4.0, 14.0)
        mc_samples[:, 2] = np.clip(mc_samples[:, 2], 0.05, 0.8)
        mc_samples[:, 3] = np.clip(mc_samples[:, 3], 0.5, 10.0)

        # Precompute initial compartment state at day D for each parameter sample
        sample_states_at_D = []
        for b_s, m_s, g_s, s_s in mc_samples:
            ktr_s = 4.0 / m_s
            if d_val > 0 and len(prior_doses) > 0 and s_s > 0:
                sol_d_s = solve_ivp(
                    fun=lambda t, y, b=b_s, k=ktr_s, g=g_s, sl=s_s: (
                        friberg_compartmental_rhs(
                            t, y, b, k, g, sl, prior_doses, self.k_decay
                        )
                    ),
                    t_span=(0.0, d_val),
                    y0=[b_s] * 5,
                    t_eval=[d_val],
                    **ODE_KW_MC,
                )
                if sol_d_s.success:
                    sample_states_at_D.append(
                        [float(sol_d_s.y[i][-1]) for i in range(5)]
                    )
                else:
                    sample_states_at_D.append([b_s] * 5)
            else:
                sample_states_at_D.append([b_s] * 5)

        results = {}
        for delay in candidate_delays:
            dose_time = d_val + float(delay)
            forward_doses = [(dose_time, float(dose_mg))]

            # Evaluate primary trajectory relative to dose: dose_time + [0, 1, ..., horizon_days]
            t_eval_dose = [dose_time + dt for dt in range(horizon_days + 1)]
            t_span = (d_val, dose_time + float(horizon_days))

            sol_primary = solve_ivp(
                lambda t, y: friberg_compartmental_rhs(
                    t=t,
                    y=y,
                    base_anc=b_val,
                    ktr=ktr_val,
                    gamma=base_gamma,
                    slope=eff_slope,
                    dose_schedule=forward_doses,
                    k_decay=self.k_decay,
                ),
                t_span,
                state_at_D,
                t_eval=t_eval_dose,
                **ODE_KW,
            )

            if sol_primary.success:
                anc_curve = np.maximum(0.1, sol_primary.y[4])
            else:
                anc_curve = np.full(horizon_days + 1, fill_value=b_val)

            # Nadir post-dose (evaluating post-infusion days 1..horizon)
            post_days = anc_curve[1:]
            post_nadir = float(np.min(post_days))
            nadir_day = int(np.argmin(post_days) + 1)
            below_count = int(np.sum(post_days < threshold))

            # Simulate each MC parameter set to obtain simulation-based P(nadir < 1.0)
            mc_nadirs = []
            mc_days_below_1 = []
            for s_idx, (b_s, m_s, g_s, s_s) in enumerate(mc_samples):
                ktr_s = 4.0 / m_s
                st_s = sample_states_at_D[s_idx]
                sol_s = solve_ivp(
                    fun=lambda t, y, b=b_s, k=ktr_s, g=g_s, sl=s_s: (
                        friberg_compartmental_rhs(
                            t, y, b, k, g, sl, forward_doses, self.k_decay
                        )
                    ),
                    t_span=t_span,
                    y0=st_s,
                    t_eval=t_eval_dose,
                    **ODE_KW_MC,
                )
                if sol_s.success:
                    p_days_s = sol_s.y[4][1:]
                    mc_nadirs.append(float(np.min(p_days_s)))
                    mc_days_below_1.append(int(np.sum(p_days_s < threshold)))
                else:
                    mc_nadirs.append(post_nadir)
                    mc_days_below_1.append(below_count)

            mc_nadirs_arr = np.array(mc_nadirs, dtype=float)
            mc_days_arr = np.array(mc_days_below_1, dtype=float)

            # Simulation-based risk: fraction of runs where nadir is below 1.0
            simulated_p_severe = float(np.mean(mc_nadirs_arr < threshold))
            nadir_band_5, nadir_band_95 = np.percentile(mc_nadirs_arr, [5, 95])
            exp_days_below_1 = float(np.mean(mc_days_arr))

            results[delay] = {
                "curve": np.round(anc_curve, 3),
                "time_days": list(range(horizon_days + 1)),
                "predicted_nadir_anc": float(np.round(post_nadir, 2)),
                "predicted_nadir_day": nadir_day,
                "days_below_1_0": below_count,
                "days_below_threshold": below_count,
                "severe_neutropenia_risk": float(np.round(simulated_p_severe, 4)),
                "simulated_severe_risk": float(np.round(simulated_p_severe, 4)),
                "nadir_band": [
                    float(np.round(nadir_band_5, 2)),
                    float(np.round(nadir_band_95, 2)),
                ],
                "nadir_lower_90": float(np.round(nadir_band_5, 2)),
                "nadir_upper_90": float(np.round(nadir_band_95, 2)),
                "expected_days_below_1_0": float(np.round(exp_days_below_1, 2)),
                "pre_dose_anc": float(np.round(anc_curve[0], 2)),
                "state_at_decision_day": [float(np.round(v, 3)) for v in state_at_D],
            }
        return results

    @staticmethod
    def combine_hybrid_toxicity_risk(
        ml_prob: float,
        predicted_nadir_anc: float,
        days_below_1_0: int,
        mechanistic_weight: float = 0.50,
        ml_weight: float = 0.50,
        simulated_severe_risk: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Synthesize a hybrid toxicity risk combining mechanistic Friberg & ML.

        Parameters
        ----------
        ml_prob : float
            Calibrated/predicted severe toxicity probability from the ML model.
        predicted_nadir_anc : float
            Forward predicted nadir ANC from the Friberg ODE trajectory.
        days_below_1_0 : int
            Forward predicted duration (days) below 1.0 x 10^9/L.
        mechanistic_weight : float
            Weight for the mechanistic component (default 0.50).
        ml_weight : float
            Weight for the ML second-opinion component (default 0.50).
        simulated_severe_risk : Optional[float]
            Monte Carlo simulation-based P(nadir < 1.0). If provided, directly
            serves as the mechanistic risk, replacing heuristic sigmoid formulas.

        Returns
        -------
        dict with keys:
            'hybrid_risk': combined risk probability in [0, 1]
            'ml_risk': ML model predicted risk
            'mech_risk': mechanistic model risk
            'concordance': 'Concordant' or 'Discordant'
        """
        if simulated_severe_risk is not None:
            mech_risk = float(np.clip(simulated_severe_risk, 0.0, 1.0))
        else:
            # Fallback continuous score from nadir when MC risk is omitted
            nadir_deficit = max(0.0, 1.0 - predicted_nadir_anc)
            duration_penalty = min(1.0, days_below_1_0 / 4.0)
            raw_mech_score = 1.0 / (1.0 + np.exp(2.8 * (predicted_nadir_anc - 0.95)))
            mech_risk = float(
                np.clip(
                    raw_mech_score + 0.25 * duration_penalty + 0.15 * nadir_deficit,
                    0.01,
                    0.99,
                )
            )

        w_sum = mechanistic_weight + ml_weight
        w_m = mechanistic_weight / w_sum
        w_ml = ml_weight / w_sum
        hybrid_p = float(np.clip(w_m * mech_risk + w_ml * ml_prob, 0.0, 1.0))

        # Concordance check (difference <= 25% points)
        diff = abs(mech_risk - ml_prob)
        concordance = "Concordant" if diff <= 0.25 else "Discordant"

        return {
            "hybrid_risk": float(np.round(hybrid_p, 4)),
            "ml_risk": float(np.round(ml_prob, 4)),
            "mech_risk": float(np.round(mech_risk, 4)),
            "risk_diff": float(np.round(diff, 4)),
            "concordance": concordance,
        }


def fit_all_patient_friberg_models(
    clinical_df: pd.DataFrame,
    profiles_df: Optional[pd.DataFrame] = None,
    wearable_df: Optional[pd.DataFrame] = None,
    decision_day: Optional[int] = None,
    strain_nudge_weight: float = 0.30,
) -> pd.DataFrame:
    """Fit patient-specific Friberg parameters for all patients in clinical_df.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Longitudinal clinical logs with patient_id, day_index, anc, dose_mg.
    profiles_df : Optional[pd.DataFrame]
        Patient demographic profiles with baseline_anc.
    wearable_df : Optional[pd.DataFrame]
        Wearable records for autonomic marrow strain estimation.
    decision_day : Optional[int]
        If set, fits strictly using day_index <= decision_day.
    strain_nudge_weight : float
        Fractional downward nudge for recovery rate under strain.

    Returns
    -------
    pd.DataFrame
        Dataframe containing patient_id, base_anc, mtt, recovery_rate,
        raw_recovery_rate, marrow_strain, drug_slope, residual_std.
    """
    model = FribergANCModel()
    results = []

    profile_map = {}
    if profiles_df is not None and "patient_id" in profiles_df.columns:
        for _, row in profiles_df.iterrows():
            profile_map[row["patient_id"]] = float(row.get("baseline_anc", 4.5))

    wearable_groups = {}
    if wearable_df is not None and "patient_id" in wearable_df.columns:
        for w_pid, w_grp in wearable_df.groupby("patient_id"):
            wearable_groups[w_pid] = w_grp

    for p_id, group in clinical_df.groupby("patient_id"):
        prior_base = profile_map.get(p_id, None)
        p_wear = wearable_groups.get(p_id, None)
        params = model.fit_patient_anc(
            patient_history=group,
            decision_day=decision_day,
            baseline_anc_prior=prior_base,
            wearable_history=p_wear,
            strain_nudge_weight=strain_nudge_weight,
        )
        params["patient_id"] = p_id
        results.append(params)

    df_out = pd.DataFrame(results)
    cols = [
        "patient_id",
        "base_anc",
        "mtt",
        "recovery_rate",
        "raw_recovery_rate",
        "marrow_strain",
        "drug_slope",
        "residual_std",
        "Base",
        "MTT",
        "gamma",
        "slope",
    ]
    return df_out[cols]
