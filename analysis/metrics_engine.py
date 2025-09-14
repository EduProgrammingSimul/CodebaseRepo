# analysis/metrics_engine.py

"""
================================================================================
          Robust Performance Metrics Engine (DTAF v3.1)
================================================================================
This final version implements the full Advanced Metrics Suite, replacing settling
time with more powerful, nuanced metrics. All calculations are hardened against
potential data errors like empty or short DataFrames.
"""

import logging
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.stats import entropy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetricsEngine:
    """Calculates performance metrics from a simulation results DataFrame. v3.1"""

    # Defines all possible metrics this engine can calculate.
    METRIC_KEYS = [
        'transient_severity_score', 'grid_load_following_index', 
        'agility_response_time_index', 'integrated_thermal_margin_violation_c_s', 
        'thermal_transient_burden', 'core_power_oscillation_index', 
        'max_rotor_angle_deviation_rad', 'negative_damping_events', 'control_policy_entropy',
        'max_overshoot_speed_pct', 'max_undershoot_speed_pct', 'iae_freq_hz_s', 'ise_freq_hz_s',
        'time_over_fuel_temp_limit_s', 'time_over_speed_limit_s', 'time_outside_freq_limit_s', 
        'total_time_unsafe_s', 'max_fuel_temp_c', 'max_speed_rpm', 'max_freq_deviation_hz', 
        'min_freq_nadir_hz', 'control_effort_valve_abs_sum', 'control_effort_valve_sq_sum', 
        'valve_reversals'
    ]

    def __init__(self, core_config: Dict[str, Any]):
        """
        Initializes the MetricsEngine.
        Args:
            core_config (Dict[str, Any]): The 'CORE_PARAMETERS' dictionary.
        """
        self.config = core_config
        self.safety_limits = self.config.get('safety_limits', {})
        logger.info("Metrics Engine v3.1 (Full Advanced Suite) initialized.")

    def calculate(self,
                  results_df: pd.DataFrame,
                  scenario_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates a dictionary of all performance metrics. Hardened against bad data.
        """
        # Initialize all metrics to NaN. They will be overwritten if successfully calculated.
        metrics: Dict[str, float] = {key: np.nan for key in self.METRIC_KEYS}
        
        # --- Robustness Check ---
        # If the simulation failed early, the DataFrame will be empty or too short.
        if results_df.empty or 'time_s' not in results_df.columns or len(results_df) < 20:
            logger.warning("Metrics calculation skipped: DataFrame is empty or too short.")
            return metrics

        try:
            time = results_df['time_s'].values
            dt = time[1] - time[0] if len(time) > 1 else 0.02
            f_nominal = self.config.get('grid', {}).get('f_nominal', 60.0)
            target_speed_rpm = self.config.get('turbine', {}).get('omega_nominal_rpm', 1800.0)
            
            # --- ADVANCED METRIC CALCULATIONS ---
            
            # Measures how well mechanical power tracks load demand. Higher is better.
            power_error = results_df['load_demand_mw'] - results_df['mechanical_power_mw']
            metrics['grid_load_following_index'] = 1000 / (1 + np.mean(np.square(power_error)))

            # Measures how quickly the turbine responds to a large, sudden load change. Lower is better.
            if 'sudden' in scenario_config.get('description', '').lower():
                load_diff = results_df['load_demand_mw'].diff().abs()
                if not load_diff.dropna().empty:
                    step_idx = load_diff.idxmax()
                    t_step = results_df.loc[step_idx, 'time_s']
                    p_initial = results_df.loc[step_idx - 1, 'mechanical_power_mw']
                    p_final_demand = results_df.loc[step_idx, 'load_demand_mw']
                    p_change_req = p_final_demand - p_initial
                    response_threshold = self.safety_limits.get('arti_response_threshold', 0.9)
                    p_response_target = p_initial + (p_change_req * response_threshold)
                    response_df = results_df[results_df['time_s'] >= t_step]
                    sign = np.sign(p_change_req)
                    response_indices = response_df.index[(response_df['mechanical_power_mw'] * sign >= p_response_target * sign)]
                    if not response_indices.empty:
                        metrics['agility_response_time_index'] = response_df.loc[response_indices[0], 'time_s'] - t_step

            # Integrated violation of the thermal warning margin. Lower is better.
            t_warn = self.safety_limits.get('max_fuel_temp_c', 2800.0) * self.safety_limits.get('fuel_temp_warning_fraction', 0.95)
            thermal_excursion = (results_df['T_fuel'] - t_warn).clip(lower=0)
            metrics['integrated_thermal_margin_violation_c_s'] = simps(thermal_excursion, time)

            # Measures the cumulative rate of change of moderator temperature. Lower is better.
            if 'T_moderator' in results_df.columns and not results_df['T_moderator'].isnull().all():
                dT_mod_dt = np.diff(results_df['T_moderator'].values) / dt
                metrics['thermal_transient_burden'] = simps(np.abs(dT_mod_dt), time[1:])

            # Measures steady-state power stability. Lower is better.
            final_power_segment = results_df['reactor_power_mw'].iloc[int(len(results_df) * 0.5):]
            metrics['core_power_oscillation_index'] = final_power_segment.std()

            # Measures rotor angle stability. Lower is better.
            if 'rotor_angle_rad' in results_df.columns and results_df['rotor_angle_rad'].notna().any():
                metrics['max_rotor_angle_deviation_rad'] = results_df['rotor_angle_rad'].max() - results_df['rotor_angle_rad'].min()

            # Counts events where the controller might be fighting the system's natural damping. Lower is better.
            omega_pu = results_df['grid_frequency_hz'] / f_nominal
            power_mismatch = results_df['mechanical_power_mw'] - results_df['load_demand_mw']
            metrics['negative_damping_events'] = ((power_mismatch > 1) & (omega_pu > 1.0001)).sum() + ((power_mismatch < -1) & (omega_pu < 0.9999)).sum()

            # Measures the unpredictability/randomness of the control policy.
            valve_pos = results_df['v_pos_actual'].dropna()
            if len(valve_pos) > 1:
                hist, _ = np.histogram(valve_pos, bins=20, range=(0, 1), density=True)
                metrics['control_policy_entropy'] = entropy(hist + 1e-9, base=2)
            
            # --- Standard & Safety Metrics ---
            metrics['max_freq_deviation_hz'] = (results_df['grid_frequency_hz'] - f_nominal).abs().max()
            metrics['min_freq_nadir_hz'] = results_df['grid_frequency_hz'].min()
            metrics['max_speed_rpm'] = results_df['speed_rpm'].max()
            metrics['max_fuel_temp_c'] = results_df['T_fuel'].max()

            final_speed = results_df['speed_rpm'].iloc[-1]
            peak_speed = metrics['max_speed_rpm']
            nadir_speed = results_df['speed_rpm'].min()
            metrics['max_overshoot_speed_pct'] = max(0.0, (peak_speed - final_speed) / target_speed_rpm * 100.0) if target_speed_rpm > 1e-6 else 0.0
            metrics['max_undershoot_speed_pct'] = max(0.0, (final_speed - nadir_speed) / target_speed_rpm * 100.0) if target_speed_rpm > 1e-6 else 0.0

            freq_error = results_df['grid_frequency_hz'] - f_nominal
            metrics['iae_freq_hz_s'] = simps(freq_error.abs(), time)
            metrics['ise_freq_hz_s'] = simps(np.square(freq_error), time)

            valve_diff = results_df['v_pos_actual'].diff()
            metrics['control_effort_valve_abs_sum'] = valve_diff.abs().sum()
            metrics['control_effort_valve_sq_sum'] = np.square(valve_diff).sum()
            metrics['valve_reversals'] = (np.diff(np.sign(valve_diff.dropna())) != 0).sum()

            # Combined score for transient severity based on speed and frequency excursions. Lower is better.
            weights = self.safety_limits.get('transient_severity_weights', {})
            freq_dev_limit = self.safety_limits.get('freq_deviation_limit_hz', 1.0)
            speed_dev_limit = self.safety_limits.get('max_speed_rpm', 2250.0) - target_speed_rpm
            max_speed_dev = metrics['max_speed_rpm'] - target_speed_rpm
            freq_severity = (metrics['max_freq_deviation_hz'] / freq_dev_limit) if freq_dev_limit > 1e-6 else 0
            speed_severity = max(0, max_speed_dev / speed_dev_limit if speed_dev_limit > 1e-6 else 0)
            metrics['transient_severity_score'] = (weights.get('w_freq_severity', 0.6) * freq_severity) + (weights.get('w_speed_severity', 0.4) * speed_severity)
            
            # --- Critical Safety Violation Times ---
            metrics['time_over_fuel_temp_limit_s'] = (results_df['T_fuel'] > self.safety_limits.get('max_fuel_temp_c', 9999)).sum() * dt
            metrics['time_over_speed_limit_s'] = (results_df['speed_rpm'] > self.safety_limits.get('max_speed_rpm', 9999)).sum() * dt
            metrics['time_outside_freq_limit_s'] = ((results_df['grid_frequency_hz'] < self.safety_limits.get('min_frequency_hz', 0)) | (results_df['grid_frequency_hz'] > self.safety_limits.get('max_frequency_hz', 99))).sum() * dt
            metrics['total_time_unsafe_s'] = metrics['time_over_fuel_temp_limit_s'] + metrics['time_over_speed_limit_s'] + metrics['time_outside_freq_limit_s']

        except Exception as e:
            logger.error(f"CRITICAL ERROR during metric calculation: {e}", exc_info=True)
            # On any error, return the dictionary of NaNs
            return {key: np.nan for key in self.METRIC_KEYS}

        return metrics
