# models/turbine_model.py

"""
================================================================================
          Robust Turbine & Governor Model (DTAF v2.2)
================================================================================
This file contains the turbine-generator model, updated to be fully compatible
with the project's "single source of truth" configuration standard and to
implement a more physically realistic dynamic response.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TurbineModel:
    """
    A simplified but robust model of the turbine and governor system, aligned
    with the DTAF v2.2 configuration standard.
    """
    def __init__(self, turbine_params: Dict[str, Any], coupling_params: Dict[str, Any]):
        """
        Initializes the TurbineModel with rigorous parameter extraction.

        Args:
            turbine_params (dict): The 'turbine' parameter dictionary from the core config.
            coupling_params (dict): The 'coupling' parameter dictionary from the core config.
        """
        logger.info("Initializing robust TurbineModel.")
        try:
            # --- Coupling Parameters (Corrected Key) ---
            self.eta_transfer = coupling_params['eta_transfer']
            self.tau_delay = coupling_params.get('tau_delay', 2.0)

            # --- Turbine-Specific Parameters ---
            self.tau_t = turbine_params['tau_t']  # Turbine mechanical time constant
            self.tau_v = turbine_params['tau_v']  # Governor valve actuator time constant
            self.omega_nominal_rpm = turbine_params['omega_nominal_rpm']

            # --- Internal State Variables ---
            self.mechanical_power: float = 0.0 # Output mechanical power (MWm)
            self.valve_position: float = 0.8 # Actual valve position (0 to 1)
            self.speed_rpm: float = self.omega_nominal_rpm # Turbine speed

            logger.info("TurbineModel initialized successfully.")

        except KeyError as e:
            logger.error(f"FATAL: Missing required key in turbine/coupling params: {e}", exc_info=True)
            raise

    def reset(self, initial_mech_power: float = 2800.0, initial_valve_pos: float = 0.8):
        """Resets the turbine to an initial state."""
        logger.info(f"Resetting TurbineModel.")
        self.mechanical_power = initial_mech_power
        self.valve_position = initial_valve_pos
        self.speed_rpm = self.omega_nominal_rpm
        logger.debug(f"Reset state: MechPower={self.mechanical_power:.2f} MW, ValvePos={self.valve_position:.3f}")

    def step(self, dt: float, thermal_power_mw: float, valve_command: float) -> float:
        """
        Advances the turbine state by one time step.

        Args:
            dt (float): The simulation time step.
            thermal_power_mw (float): The current thermal power from the reactor (in MWth).
            valve_command (float): The commanded valve position from the controller [0, 1].

        Returns:
            float: The updated mechanical power output (in MWm).
        """
        # 1. Model the governor valve actuator lag (first-order system)
        dv_dt = (1 / self.tau_v) * (valve_command - self.valve_position)
        self.valve_position += dv_dt * dt
        self.valve_position = np.clip(self.valve_position, 0.0, 1.0)

        # 2. Calculate the steam power available at the turbine inlet
        # This is throttled by the valve position.
        effective_steam_power = self.valve_position * (self.eta_transfer * thermal_power_mw)
        
        # 3. Model the turbine mechanical power response (first-order system)
        dp_mech_dt = (1 / self.tau_t) * (effective_steam_power - self.mechanical_power)
        self.mechanical_power += dp_mech_dt * dt
        
        # Note: The turbine speed (speed_rpm) is not calculated here.
        # It is calculated in the GridModel based on the power imbalance,
        # which is a more accurate representation of the system dynamics.
        # This model's primary output is the mechanical power.

        logger.debug(f"Turbine step: V_cmd={valve_command:.3f}, V_act={self.valve_position:.3f}, P_mech={self.mechanical_power:.2f} MW")
        
        return self.mechanical_power
