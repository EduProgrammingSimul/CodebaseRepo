# environment/reactor_controller.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReactorController:
    """
    A simple, tuned PI controller to manage reactor power by adjusting control rod
    reactivity. Its goal is to maintain the average moderator temperature
    at a specified setpoint.
    """
    def __init__(self, dt: float):
        # --- TUNED GAINS (CRITICAL FIX) ---
        # The original gains were too aggressive for the sensitive point-kinetics
        # model, causing oscillations and runaway power. These values are
        # significantly reduced to ensure a smooth, stable response that
        # prioritizes stability over rapid correction, which is realistic for
        # a large thermal system like a reactor core.
        self.kp = 0.00008  # Proportional gain (significantly reduced)
        self.ki = 0.00002  # Integral gain (significantly reduced)
        
        self.dt = dt
        self.setpoint = 306.5  # Target T_moderator in Celsius
        
        self._integral = 0.0
        # Reactivity limits are kept to prevent excessive insertion rates
        self._reactivity_limits = (-0.005, 0.005)

    def reset(self, setpoint: float = 306.5):
        """Resets the controller's integral term and setpoint."""
        self._integral = 0.0
        self.setpoint = setpoint
        logger.debug(f"ReactorController reset with setpoint: {self.setpoint:.2f} C")

    def step(self, current_moderator_temp: float) -> float:
        """
        Calculates the required rod_reactivity adjustment.

        Args:
            current_moderator_temp (float): The current T_moderator from the reactor model.

        Returns:
            float: The calculated control rod reactivity.
        """
        if self.dt <= 0:
            return 0.0

        error = self.setpoint - current_moderator_temp
        
        # Integral term with anti-windup
        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -5.0, 5.0) # Integral clamp
        
        # Calculate final reactivity
        reactivity_output = (self.kp * error) + (self.ki * self._integral)
        
        # Clip to safe operational limits
        return float(np.clip(reactivity_output, self._reactivity_limits[0], self._reactivity_limits[1]))
