# analysis/results_analyzer.py

"""
================================================================================
          Robust Results Analyzer for KPI Calculation (DTAF v2.2)
================================================================================
This utility contains a simplified function for calculating a core set of Key
Performance Indicators (KPIs) from simulation results.

This enhanced version includes robust data validation and safe calculation
methods to prevent errors when processing data from potentially failed
simulation runs.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_kpis(results_df: pd.DataFrame, target_speed_rpm: float) -> Dict[str, float]:
    """
    Calculates Key Performance Indicators (KPIs) from simulation results.

    This function is designed to be robust and will return NaN for any KPI
    that cannot be calculated due to missing or invalid data, rather than crashing.

    Args:
        results_df (pd.DataFrame): DataFrame with simulation results.
                                   Must include 'time_s' and 'turbine_speed_rpm'.
        target_speed_rpm (float): The target turbine speed in RPM for error calculation.

    Returns:
        Dict[str, float]: A dictionary of calculated KPIs.
    """
    # Initialize all KPIs to NaN to ensure a consistent return structure
    kpis = {
        'iae_speed_rpm_s': np.nan,
        'overshoot_speed_percent': np.nan,
    }

    # --- 1. Robustness Check: Validate input DataFrame ---
    if results_df.empty:
        logger.warning("KPI Calculation failed: Input DataFrame is empty.")
        return kpis
    if 'time_s' not in results_df.columns or 'turbine_speed_rpm' not in results_df.columns:
        logger.warning(f"KPI Calculation failed: DataFrame is missing required columns ('time_s', 'turbine_speed_rpm').")
        return kpis
    if len(results_df) < 2:
        logger.warning("KPI Calculation failed: At least 2 data points are required.")
        return kpis

    try:
        # --- 2. Data Preparation ---
        df = results_df.sort_values('time_s').reset_index(drop=True)
        dt = df['time_s'].diff().fillna(0).values
        speed = df['turbine_speed_rpm'].values

        # --- 3. IAE Calculation (Integral Absolute Error) ---
        speed_error = speed - target_speed_rpm
        # Use nansum to safely ignore potential NaN values in the data
        kpis['iae_speed_rpm_s'] = np.nansum(np.abs(speed_error) * dt)

        # --- 4. Overshoot Calculation ---
        # Use nanmax to find the peak speed, ignoring NaNs
        max_speed = np.nanmax(speed)
        if pd.notna(max_speed) and abs(target_speed_rpm) > 1e-6:
            overshoot_value = max_speed - target_speed_rpm
            # Overshoot is only considered if the max speed exceeds the target
            kpis['overshoot_speed_percent'] = max(0.0, (overshoot_value / target_speed_rpm) * 100.0)
        else:
            kpis['overshoot_speed_percent'] = 0.0 # No overshoot if max speed is NaN or <= target

        logger.info(f"KPIs calculated successfully: {kpis}")

    except Exception as e:
        logger.error(f"Error during KPI calculation: {e}", exc_info=True)
        # On any unexpected error, return the dict of NaNs
        return {key: np.nan for key in kpis}

    return kpis