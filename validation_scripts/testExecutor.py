# testExecutor.py

"""
================================================================================
          Standalone Scenario Executor for Validation (DTAF v3.x)
================================================================================
Purpose:
This script provides a focused, robust, and standalone method to test the core 
simulation workflow. It is designed specifically to validate that the critical 
bug related to the ScenarioExecutor and PWRGymEnvUnified initialization has been 
resolved.

It performs the following steps:
1.  Sets up the Python path to recognize the project modules.
2.  Loads the master configuration from 'config/parameters.py'.
3.  Selects a single, simple scenario ('baseline_steady_state') for the test.
4.  Loads a single, standard controller ('PID').
5.  Instantiates the ScenarioExecutor using the full, correct configuration.
6.  Executes the simulation.
7.  Performs rigorous checks on the output to confirm success:
    - Verifies that the resulting DataFrame is not empty.
    - Calculates and displays all performance metrics using the MetricsEngine.
8.  Provides clear, detailed logging of its success or failure.
"""

import logging
import os
import sys
import pandas as pd
from typing import Dict, Any

# --- 1. Set up project path and logging ---
def setup_environment():
    """Configures the Python path and logging for the test run."""
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Configure logging to show detailed info from the framework components
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress overly verbose logs from matplotlib if it's used by other modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return project_root

# --- 2. Main Test Execution Logic ---
def run_validation_test(project_root: str):
    """
    Orchestrates the validation test.
    """
    logger = logging.getLogger("ValidationTest")
    logger.info("=" * 70)
    logger.info("  STANDALONE SIMULATION WORKFLOW VALIDATION TEST STARTING")
    logger.info("=" * 70)

    try:
        # --- Import framework components AFTER setting the path ---
        from analysis.parameter_manager import ParameterManager
        from analysis.scenario_executor import ScenarioExecutor
        from analysis.metrics_engine import MetricsEngine
        from analysis.scenario_definitions import get_scenarios
        from controllers import load_controller

        # --- Test Parameters ---
        CONFIG_FILE = os.path.join(project_root, 'config', 'parameters.py')
        TEST_SCENARIO_NAME = 'baseline_steady_state'
        TEST_CONTROLLER_NAME = 'PID' # Using the basic, reliable PID controller

        # --- Load Master Configuration ---
        logger.info(f"Loading master configuration from: {CONFIG_FILE}")
        param_manager = ParameterManager(config_filepath=CONFIG_FILE)
        full_config = param_manager.get_all_parameters()
        core_config = full_config.get('CORE_PARAMETERS')
        if not core_config:
            raise ValueError("CRITICAL: 'CORE_PARAMETERS' not found in config file.")
        
        # --- Load Scenario and Controller ---
        logger.info(f"Preparing to test scenario: '{TEST_SCENARIO_NAME}'")
        all_scenarios = get_scenarios(core_config)
        test_scenario_config = all_scenarios.get(TEST_SCENARIO_NAME)
        if not test_scenario_config:
            raise ValueError(f"CRITICAL: Test scenario '{TEST_SCENARIO_NAME}' not found in definitions.")

        logger.info(f"Loading test controller: '{TEST_CONTROLLER_NAME}'")
        sim_dt = core_config.get('simulation', {}).get('dt', 0.02)
        controller_instance, _ = load_controller(TEST_CONTROLLER_NAME, full_config, sim_dt)
        if not controller_instance:
            raise ValueError(f"CRITICAL: Failed to load controller '{TEST_CONTROLLER_NAME}'.")
            
        # --- Instantiate Core Simulation and Analysis Engines ---
        logger.info("Instantiating ScenarioExecutor...")
        # This is the critical step: The executor now receives the full config,
        # which it will use to correctly instantiate the environment.
        executor = ScenarioExecutor(base_env_config_full=full_config)
        
        logger.info("Instantiating MetricsEngine...")
        metrics_engine = MetricsEngine(core_config=core_config)

        # --- Execute the Simulation ---
        logger.info("-" * 70)
        logger.info(f"Executing simulation for '{TEST_SCENARIO_NAME}' with '{TEST_CONTROLLER_NAME}'...")
        start_time = pd.Timestamp.now()
        
        results_df = executor.execute(
            scenario_name=TEST_SCENARIO_NAME,
            scenario_config_from_caller=test_scenario_config,
            controller_name=TEST_CONTROLLER_NAME,
            controller_instance=controller_instance
        )
        
        duration = pd.Timestamp.now() - start_time
        logger.info(f"Execution completed in: {duration.total_seconds():.2f} seconds.")
        logger.info("-" * 70)
        
        # --- 3. Validate the Results ---
        logger.info("Performing results validation...")
        if results_df.empty or len(results_df) < 2:
            logger.error("************************************************************")
            logger.error("  VALIDATION FAILED: The simulation produced an empty or")
            logger.error("  insufficient DataFrame. The root cause likely persists.")
            logger.error("************************************************************")
            return

        logger.info("✅ SUCCESS: Simulation produced a non-empty DataFrame.")
        print("\n--- DataFrame Info ---")
        results_df.info()
        print("\n--- DataFrame Head ---")
        print(results_df.head())
        print("\n--- DataFrame Tail ---")
        print(results_df.tail())
        
        logger.info("\nCalculating performance metrics...")
        metrics = metrics_engine.calculate(results_df, test_scenario_config)
        
        if not metrics or pd.isna(list(metrics.values())).all():
            logger.error("************************************************************")
            logger.error("  VALIDATION FAILED: Metrics could not be calculated.")
            logger.error("  The DataFrame might contain NaN or unexpected values.")
            logger.error("************************************************************")
            return

        logger.info("✅ SUCCESS: Performance metrics calculated successfully.")
        print("\n--- Calculated Metrics ---")
        for key, value in metrics.items():
            if pd.notna(value):
                print(f"  - {key:<40}: {value:.4f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("  VALIDATION TEST PASSED: The core simulation workflow is operational.")
        logger.info("=" * 70)

    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during the test execution: {e}", exc_info=True)
        logger.info("\n" + "=" * 70)
        logger.info("  VALIDATION TEST FAILED due to a critical exception.")
        logger.info("=" * 70)

# --- 4. Script Entry Point ---
if __name__ == "__main__":
    project_root_path = setup_environment()
    run_validation_test(project_root_path)