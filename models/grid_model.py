# models/grid_model.py

"""
================================================================================
          Robust Grid Physics Model (DTAF v2.2)
================================================================================
This file contains the electrical grid model, updated to be fully compatible
with the project's "single source of truth" configuration standard and to
implement the standard swing equation for frequency dynamics.
"""

import numpy as np
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class GridModel:
    """
    A model for the electrical grid, simulating frequency dynamics based on
    the power balance using the swing equation.
    """
    def __init__(self, grid_params: Dict[str, Any], sim_params: Dict[str, Any]):
        """
        Initializes the GridModel.

        Args:
            grid_params (dict): The 'grid' parameter dictionary from the core config.
            sim_params (dict): The 'simulation' parameter dictionary.
        """
        logger.info("Initializing robust GridModel.")
        try:
            # --- Swing Equation Parameters ---
            self.H = grid_params['H']  # Inertia constant (s)
            self.D = grid_params['D']  # Damping coefficient (p.u.)
            self.f_nominal = grid_params['f_nominal']  # Nominal frequency (Hz)
            self.S_base = grid_params['S_base'] # System MVA base
            
            # --- Internal State Variables ---
            self.frequency: float = self.f_nominal
            self.omega_pu: float = 1.0 # Speed in per-unit
            self.delta: float = 0.0 # Rotor angle (rad)
            self.current_demand: float = 0.0 # Electrical load demand (MW)
            self.load_profile_func: Optional[Callable[[float, int], float]] = None

            logger.info("GridModel initialized successfully.")

        except KeyError as e:
            logger.error(f"FATAL: Missing required key in grid_params: {e}", exc_info=True)
            raise

    def reset(self, initial_load_mw: float):
        """Resets the grid to its initial state."""
        logger.info(f"Resetting GridModel with initial load: {initial_load_mw:.2f} MW")
        self.frequency = self.f_nominal
        self.omega_pu = 1.0
        self.delta = 0.0
        self.current_demand = initial_load_mw
        # Default to a constant load if no dynamic profile is set
        self.load_profile_func = lambda time_s, step: initial_load_mw

    def set_load_profile(self, load_func: Callable[[float, int], float]):
        """Sets a dynamic load profile function for the simulation."""
        self.load_profile_func = load_func
        logger.info("Dynamic load profile has been set for the GridModel.")

    def step(self, dt: float, mechanical_power_mw: float, time_s: float, step_num: int):
        """
        Advances the grid state by one time step using the swing equation.

        Args:
            dt (float): The simulation time step.
            mechanical_power_mw (float): The mechanical power from the turbine (MWm).
            time_s (float): The current simulation time in seconds.
            step_num (int): The current simulation step number.
        """
        # 1. Update the electrical load demand from the profile function
        self.current_demand = self.load_profile_func(time_s, step_num)

        # 2. Convert powers to per-unit (p.u.) for the swing equation
        p_m_pu = mechanical_power_mw / self.S_base
        p_e_pu = self.current_demand / self.S_base

        # 3. Solve the Swing Equation (d(omega)/dt part)
        # d(omega_pu)/dt = (1 / 2H) * (P_m - P_e - D * (omega_pu - 1))
        d_omega_pu_dt = (1 / (2 * self.H)) * (p_m_pu - p_e_pu - self.D * (self.omega_pu - 1.0))
        
        # Update speed in per-unit
        self.omega_pu += d_omega_pu_dt * dt
        
        # 4. Update frequency in Hz
        self.frequency = self.omega_pu * self.f_nominal

        # 5. Solve for rotor angle (optional, but good for completeness)
        # d(delta)/dt = (omega_pu - 1) * 2 * pi * f_nominal
        d_delta_dt = (self.omega_pu - 1.0) * 2 * np.pi * self.f_nominal
        self.delta += d_delta_dt * dt

        logger.debug(f"Grid step: Freq={self.frequency:.4f} Hz, P_mech={mechanical_power_mw:.2f} MW, P_elec={self.current_demand:.2f} MW")

