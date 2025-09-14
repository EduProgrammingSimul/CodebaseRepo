# models/reactor_model.py

"""
================================================================================
          Robust Reactor Physics Model (DTAF v2.2)
================================================================================
This file contains the reactor physics model, updated to be fully compatible
with the project's "single source of truth" configuration standard.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReactorModel:
    """
    Implements a point kinetics reactor model with thermal feedback, aligned
    with the DTAF v2.2 configuration standard.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the reactor model with rigorous parameter extraction.

        Args:
            params (dict): The reactor parameter dictionary from the core config.
        """
        logger.info("Initializing robust ReactorModel.")
        try:
            # --- Point Kinetics Parameters (Corrected Keys) ---
            self.beta_i = np.array(params['beta_i'])
            self.lambda_i = np.array(params['lambda_i'])
            self.Lambda = params['Lambda']  # Prompt neutron generation time
            self.beta_total = params['beta_total']

            # --- Reactivity Feedback Coefficients (Corrected Keys) ---
            self.alpha_f = params['alpha_f']  # Fuel temp coefficient
            self.alpha_c = params['alpha_c']  # Coolant temp coefficient

            # --- Thermal-Hydraulic Parameters (Corrected Keys) ---
            self.C_f = params['C_f']  # Fuel Heat Capacity
            self.C_c = params['C_c']  # Coolant Heat Capacity
            self.Omega = params['Omega'] # Fuel-to-Coolant Heat Transfer Coefficient

            # --- Nominal & Initial Conditions ---
            self.P0 = params['P0'] # Nominal Full Thermal Power (MWth)
            self.T_inlet = params['T_inlet']
            self.T_coolant0 = params['T_coolant0']
            self.T_fuel0 = params['T_fuel0']
            
            # --- Internal State Variables ---
            self.power_level: float = 0.0
            self.precursor_concentrations: np.ndarray = np.zeros_like(self.beta_i)
            self.T_fuel: float = 0.0
            self.T_moderator: float = 0.0 # Using T_moderator for consistency with older API if needed

            logger.info("ReactorModel initialized successfully.")

        except KeyError as e:
            logger.error(f"FATAL: Missing required key in reactor_params: {e}", exc_info=True)
            raise  # Re-raise to halt execution if config is invalid

    def reset(self, initial_power_fraction: float = 1.0):
        """Resets the model to a specified initial power level."""
        logger.info(f"Resetting ReactorModel to {initial_power_fraction*100:.1f}% power.")
        
        self.power_level = initial_power_fraction
        # Assume initial temperatures scale linearly for simplicity (or use lookup table)
        self.T_fuel = self.T_fuel0 * initial_power_fraction
        self.T_moderator = self.T_coolant0 * initial_power_fraction
        
        # Assume precursors are in equilibrium at the initial power level
        if self.Lambda > 1e-9: # Avoid division by zero
            self.precursor_concentrations = (self.beta_i / (self.lambda_i * self.Lambda)) * self.power_level
        else:
            self.precursor_concentrations.fill(0.0)

        logger.debug(f"Reset state: Power={self.power_level:.3f}, T_fuel={self.T_fuel:.2f}C")

    def step(self, dt: float, rod_reactivity: float) -> float:
        """
        Advances the reactor state by one time step.

        Args:
            dt (float): The simulation time step.
            rod_reactivity (float): The reactivity inserted by control rods for this step.

        Returns:
            float: The updated thermal power level in MWth.
        """
        # Calculate reactivity from temperature feedback
        delta_T_f = self.T_fuel - self.T_fuel0
        delta_T_c = self.T_moderator - self.T_coolant0
        rho_feedback = self.alpha_f * delta_T_f + self.alpha_c * delta_T_c
        
        # Total reactivity
        total_reactivity = rho_feedback + rod_reactivity

        # --- Solve Point Kinetics Equations (Euler method) ---
        lambda_c_sum = np.sum(self.lambda_i * self.precursor_concentrations)
        
        # d(Power)/dt
        dp_dt = ((total_reactivity - self.beta_total) / self.Lambda) * self.power_level + lambda_c_sum
        self.power_level += dp_dt * dt
        
        # d(Precursors)/dt
        dc_dt = (self.beta_i / self.Lambda) * self.power_level - self.lambda_i * self.precursor_concentrations
        self.precursor_concentrations += dc_dt * dt

        # --- Solve Thermal-Hydraulic Equations (Lumped model) ---
        # Power is generated as a fraction of nominal full power P0
        generated_power_mw = self.power_level * self.P0

        # dT(Fuel)/dt
        dtf_dt = (1 / self.C_f) * (generated_power_mw - self.Omega * (self.T_fuel - self.T_moderator))
        self.T_fuel += dtf_dt * dt

        # dT(Coolant)/dt - Assuming T_moderator is average coolant temp
        # A more complex model would consider coolant mass flow rate.
        # This simplified model is based on the provided parameters.
        dtc_dt = (1 / self.C_c) * (self.Omega * (self.T_fuel - self.T_moderator)) # Simplified energy transfer
        self.T_moderator += dtc_dt * dt

        # Ensure non-negative power
        self.power_level = max(0.0, self.power_level)
        
        logger.debug(f"Reactor step: P={self.power_level * self.P0:.2f} MW, Rho={total_reactivity*1e5:.2f} pcm")
        
        return self.power_level * self.P0 # Return power in MWth
