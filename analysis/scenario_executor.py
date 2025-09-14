# analysis/scenario_executor.py

"""
================================================================================
          Robust Scenario Executor for Simulation (DTAF v3.2)
================================================================================
This final, hardened version correctly instantiates the simulation environment
with all required arguments, ensuring full compatibility with the latest
environment updates and advanced metrics suite.
"""

import logging
import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, Any, Generator

from environment.pwr_gym_env import PWRGymEnvUnified
from analysis.scenario_definitions import get_scenarios

logger = logging.getLogger(__name__)

class ScenarioExecutor:
    """Executes a single simulation scenario using PWRGymEnvUnified v3.0."""

    def __init__(self, base_env_config_full: Dict[str, Any]):
        """
        Initializes the ScenarioExecutor.

        Args:
            base_env_config_full (Dict[str, Any]): The full configuration dictionary,
                                                  which MUST contain the 'CORE_PARAMETERS' key.
        """
        if 'CORE_PARAMETERS' not in base_env_config_full:
            raise ValueError("Invalid config for ScenarioExecutor: must contain 'CORE_PARAMETERS'.")
            
        self.full_config = base_env_config_full
        self.core_params = self.full_config['CORE_PARAMETERS']
        self.all_scenario_definitions = get_scenarios(self.core_params)
        logger.info("Scenario Executor v3.2 (Final Constructor) initialized.")

    def execute(self,
                scenario_name: str,
                scenario_config_from_caller: Dict[str, Any],
                controller_name: str,
                controller_instance: Any) -> pd.DataFrame:
        """
        Executes a simulation scenario and returns the results as a DataFrame.
        This is a batch-style execution method.
        """
        results_data = []
        try:
            # The yield-based executor handles the detailed step-by-step logic
            for log_entry in self.execute_and_yield(scenario_name, scenario_config_from_caller, controller_name, controller_instance):
                # Check for an error flag from the generator
                if log_entry is None or 'error' in log_entry:
                    logger.warning(f"Execution yielded an error or None for {scenario_name}/{controller_name}. Terminating collection.")
                    break
                results_data.append(log_entry)
        except Exception as e:
             logger.error(f"Unhandled exception during batch execution for {scenario_name}/{controller_name}: {e}", exc_info=True)
        
        if not results_data:
            logger.warning(f"No data was generated for {scenario_name}/{controller_name}.")
            return pd.DataFrame()
            
        return pd.DataFrame(results_data)

    def execute_and_yield(self,
                          scenario_name: str,
                          scenario_config_from_caller: Dict[str, Any],
                          controller_name: str,
                          controller_instance: Any
                          ) -> Generator[Optional[Dict[str, Any]], None, None]:
        """
        Executes a scenario step-by-step, yielding the info dictionary at each step.
        This is the core execution loop.
        """
        logger.info(f"--- Starting Validation Execution: '{scenario_name}' / '{controller_name}' ---")
        env: Optional[PWRGymEnvUnified] = None

        try:
            # CRITICAL FIX: Collect all required parameters from the core configuration
            # and pass them as keyword arguments to the environment constructor. This
            # resolves the initialization error.
            env_params = {
                'reactor_params': self.core_params.get('reactor', {}),
                'turbine_params': self.core_params.get('turbine', {}),
                'grid_params': self.core_params.get('grid', {}),
                'coupling_params': self.core_params.get('coupling', {}),
                'sim_params': self.core_params.get('simulation', {}),
                'safety_limits': self.core_params.get('safety_limits', {}),
                'rl_normalization_factors': self.core_params.get('rl_normalization_factors', {}),
                'all_scenarios_definitions': self.all_scenario_definitions,
                'initial_scenario_name': scenario_name,
                'is_training_env': False, # This is a validation/analysis run, not training
                'rl_training_config': self.core_params.get('rl_training_adv', {})
            }
            env = PWRGymEnvUnified(**env_params)
            
            reset_options = scenario_config_from_caller.get('reset_options', {})
            normalized_obs, info = env.reset(options=reset_options)
            
            # Yield the initial state before the first step
            yield {'step': -1, **info}

        except Exception as e:
            logger.error(f"Failed to initialize/reset environment for '{scenario_name}': {e}", exc_info=True)
            yield {'error': f'Env Init/Reset Failed: {e}'}
            return

        terminated, truncated = False, False
        step_count = 0
        
        # Determine the maximum number of steps for this specific scenario
        max_steps_from_scenario = scenario_config_from_caller.get('max_steps')
        max_steps = max_steps_from_scenario or self.core_params.get('simulation', {}).get('max_steps', 5000)

        while not terminated and not truncated and step_count < max_steps:
            try:
                # Get action from the controller (PID, FLC, RL, etc.)
                action_value = controller_instance.step(normalized_obs)
                action = np.array([action_value]).flatten()
            except Exception as e:
                 logger.error(f"Error getting action from controller {controller_name} at step {step_count}: {e}", exc_info=True)
                 action = np.array([0.5]) # Default to a neutral action on controller error

            # Take a step in the environment
            normalized_obs, reward, terminated, truncated, info = env.step(action)
            
            # Yield the results of the step
            yield {'step': step_count, **info}
            step_count += 1
        
        # Ensure the environment is properly closed
        env.close()
