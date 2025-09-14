# optimization_suite/optimization_utils.py

"""
================================================================================
          Robust Optimization Utilities (DTAF v2.2)
================================================================================
This file provides utility functions used by the optimization and validation
workflows, primarily for running single simulation instances and extracting
detailed metrics in a robust manner.

This enhanced version includes comprehensive error handling to ensure that
simulation failures are caught and reported correctly, preventing the optimization
process from crashing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Import project-specific modules
from analysis.scenario_executor import ScenarioExecutor
from analysis.metrics_engine import MetricsEngine

logger = logging.getLogger(__name__)

def run_single_sim_and_extract_detailed_metrics(
    base_core_config: Dict[str, Any],
    scenario_config: Dict[str, Any],
    controller_name: str,
    controller_instance: Any
) -> Dict[str, Any]:
    """
    Runs a single simulation for a given controller and scenario, calculates
    detailed metrics, and returns a structured dictionary with the results.

    This function is designed to be robust and will not crash on simulation
    failure. Instead, it will report the failure status.

    Args:
        base_core_config (dict): The CORE_PARAMETERS dictionary.
        scenario_config (dict): The configuration for the specific scenario to run.
        controller_name (str): The name of the controller being tested.
        controller_instance (Any): The instantiated controller object.

    Returns:
        Dict[str, Any]: A dictionary containing the success status, metrics,
                        raw results DataFrame, and any error messages.
    """
    scenario_display_name = scenario_config.get('name', 'UnnamedScenario')
    logger.info(f"--- Running Sim for Metrics: {scenario_display_name} / {controller_name} ---")

    # Initialize the structured output dictionary
    output_results = {
        'completed_successfully': False,
        'termination_reason': 'Unknown',
        'metrics': {},
        'raw_results_df': pd.DataFrame(),
        'error_message': None
    }

    try:
        # 1. Initialize necessary components for the run
        # The executor needs the full config structure containing CORE_PARAMETERS
        full_config_for_executor = {'CORE_PARAMETERS': base_core_config}
        executor = ScenarioExecutor(base_env_config_full=full_config_for_executor)
        metrics_engine = MetricsEngine(base_core_config.get('metrics_config', {}))

        # Reset the controller's internal state before the run
        if hasattr(controller_instance, 'reset'):
            controller_instance.reset()

        # 2. Execute the simulation
        results_df = executor.execute(
            scenario_name=scenario_display_name,
            scenario_config_from_caller=scenario_config,
            controller_name=controller_name,
            controller_instance=controller_instance
        )
        output_results['raw_results_df'] = results_df

        # 3. Process the results
        if results_df.empty or len(results_df) < 2:
            logger.warning(f"Sim for '{scenario_display_name}/{controller_name}' returned empty/insufficient data.")
            output_results['error_message'] = "Simulation returned no valid data."
            # Attempt to get a more specific reason if possible (e.g., from environment attributes if they exist)
            output_results['termination_reason'] = "Insufficient Data / Early Failure"
            return output_results

        logger.debug(f"Sim for '{scenario_display_name}/{controller_name}' successful. Results length: {len(results_df)}")
        
        # 4. Calculate metrics from the successful run
        safety_limits = base_core_config.get('safety_limits', {})
        metrics_dict = metrics_engine.calculate(results_df, safety_limits, scenario_config)
        output_results['metrics'] = metrics_dict

        # 5. Determine the final status
        # A run is only considered "completed successfully" if it did not hit a hard safety limit.
        # Reaching max_steps (truncation) is considered a successful completion.
        # The metrics engine provides the necessary data to infer this.
        if metrics_dict.get('total_time_unsafe_s', 0.0) > 1e-6:
             output_results['completed_successfully'] = False
             output_results['termination_reason'] = "Safety Limit Violation"
             logger.warning(f"Scenario '{scenario_display_name}' for '{controller_name}' deemed failed due to safety violations.")
        else:
             output_results['completed_successfully'] = True
             output_results['termination_reason'] = "Completed Nominally" # Or "Truncated" if that info is available

    except Exception as e:
        logger.error(f"Unhandled exception in run_single_sim_and_extract_detailed_metrics for '{scenario_display_name}/{controller_name}': {e}", exc_info=True)
        output_results['error_message'] = str(e)
        output_results['completed_successfully'] = False
        output_results['termination_reason'] = "Exception during execution"
    
    finally:
        logger.info(f"--- Finished Sim for Metrics: {scenario_display_name} / {controller_name}. Success: {output_results['completed_successfully']} ---")
        return output_results