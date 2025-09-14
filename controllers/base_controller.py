# controllers/base_controller.py

"""
================================================================================
          Abstract Base Class for Controllers (DTAF v2.2)
================================================================================
This file defines the abstract interface that all controller implementations
in this project must adhere to. This ensures that any controller (PID, FLC, RL)
is interchangeable and can be seamlessly used by the simulation and analysis
frameworks (ScenarioExecutor, UI, etc.).
"""

import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any

# Get a logger for this module
logger = logging.getLogger(__name__)

class BaseController(ABC):
    """
    Abstract Base Class for all controllers (PID, FLC, RL Wrappers, etc.). v2.1

    Defines the common interface required by the ScenarioExecutor and UI
    for stepping, resetting, and potentially live parameter updates.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any], dt: float):
        """
        Initialize the controller.

        Args:
            config (dict): Controller-specific configuration parameters loaded
                           (e.g., from parameters.py or optimized_controllers/).
            dt (float): Simulation time step (seconds), needed for discrete controllers.
        """
        # --- Rigorous Initialization Validation ---
        if not isinstance(config, dict):
             logger.error("Initialization failed: Controller config must be a dictionary.")
             raise TypeError("Controller config must be a dictionary.")
        if not isinstance(dt, (float, int)) or dt <= 0:
             logger.error(f"Initialization failed: Time step dt must be a positive number, got {dt}.")
             raise ValueError("Time step dt must be a positive number.")

        self.config = config
        self.dt = dt
        
        # Log initialization in the base class for traceability
        logger.info(f"Initializing BaseController implementation: {self.__class__.__name__}")
        logger.debug(f"  > Config provided: {config}")
        logger.debug(f"  > Time step (dt): {dt}")

    @abstractmethod
    def step(self, observation: np.ndarray) -> float:
        """
        Calculate the control action based on the current environment observation.

        Args:
            observation (np.ndarray): The current environment observation vector.
                                     The specific indices depend on the environment definition.

        Returns:
            float: The calculated control action (e.g., valve position command),
                   typically expected to be within a defined range (e.g., 0.0 to 1.0).
                   Must return a scalar float.
        """
        # Abstract method: Implementation is required by all concrete controller subclasses.
        pass

    @abstractmethod
    def reset(self):
        """
        Reset any internal states of the controller.

        This includes, for example, the integral term for a PID controller,
        the last recorded error for FLC/PID, or hidden states for recurrent
        RL policies if applicable. This method is called at the beginning of
        each new simulation episode to ensure a clean start.
        """
        # Base implementation logs the reset action for traceability.
        logger.info(f"Resetting controller state for: {self.__class__.__name__}")
        # Subclasses should call super().reset() if they override this,
        # or simply implement their own state-resetting logic.
        pass

    @abstractmethod
    def update_parameters(self, new_params: Dict[str, Any]):
        """
        Updates the controller's tunable internal parameters live during a run.
        This enables features like interactive tuning from the UI.

        Args:
            new_params (dict): A dictionary where keys are parameter names
                               (e.g., 'kp', 'error_scaling') and values are the
                               new settings. The controller implementation is
                               responsible for validating and applying these changes.
        """
        # Base implementation logs the attempt for traceability.
        logger.info(f"Attempting to update parameters for {self.__class__.__name__} with: {new_params}")
        # Subclasses should call super().update_parameters() before handling
        # their specific parameter updates.
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns the current values of the controller's tunable parameters.
        This is used by the UI to display current settings and to populate
        editing widgets with the correct initial values.

        Returns:
            dict: A dictionary of the controller's current tunable parameters.
                  Keys should match those expected by `update_parameters`.
        """
        # Base implementation logs the action for traceability.
        logger.info(f"Getting parameters for {self.__class__.__name__}")
        # Subclasses must implement this to return their specific parameters.
        pass