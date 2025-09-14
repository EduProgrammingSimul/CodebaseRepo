# test_rl_trainer.py

"""
================================================================================
          Standalone RL Training Workflow Validator (DTAF v3.x)
================================================================================
Purpose:
This script provides a focused "smoke test" for the entire Reinforcement
Learning training pipeline. It is designed to run very quickly and validate that
all components of the RL stack are working together correctly before launching a
full, time-consuming training session.

It performs the following steps:
1.  Sets up the Python path and logging.
2.  Loads the master configuration from 'config/parameters.py'.
3.  **Overrides key training parameters** for a minimal test run (e.g., very few
    timesteps, low evaluation frequency).
4.  Instantiates the RLTrainer with this temporary configuration.
5.  Executes the trainer.train() method.
6.  **Validates success** by checking if the trainer completes without exceptions
    and if a final model file (.zip) is created in the specified directory.
"""

import logging
import os
import sys
import copy
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
    # Suppress overly verbose logs from libraries like matplotlib or others
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return project_root

# --- 2. Main Test Execution Logic ---
def run_rl_training_test(project_root: str):
    """
    Orchestrates the RL training validation test.
    """
    logger = logging.getLogger("RL_ValidationTest")
    logger.info("=" * 70)
    logger.info("      STANDALONE RL TRAINING WORKFLOW VALIDATION TEST STARTING")
    logger.info("=" * 70)

    try:
        # --- Import framework components AFTER setting the path ---
        from analysis.parameter_manager import ParameterManager
        from optimization_suite.rl_trainer import RLTrainer

        # --- Test Configuration ---
        CONFIG_FILE = os.path.join(project_root, 'config', 'parameters.py')
        TEST_SAVE_DIR = os.path.join(project_root, "results", "test_rl_run")

        # --- Load Master Configuration ---
        logger.info(f"Loading master configuration from: {CONFIG_FILE}")
        param_manager = ParameterManager(config_filepath=CONFIG_FILE)
        full_config = param_manager.get_all_parameters()
        
        # --- Create a temporary config for the test run (CRITICAL STEP) ---
        test_config = copy.deepcopy(full_config)
        rl_params = test_config['CORE_PARAMETERS']['rl_training_adv']
        
        logger.warning("Overriding RL training parameters for a quick smoke test...")
        
        # Use a very small number of timesteps to make the test run fast.
        rl_params['total_timesteps'] = 500
        # Start learning very early.
        rl_params['learning_starts'] = 100
        # Set a temporary save path for the test model.
        rl_params['model_save_path'] = TEST_SAVE_DIR
        # Ensure the evaluation callback runs at least once during this short test.
        rl_params['eval_freq'] = 200
        # Use a smaller buffer to reduce memory usage for the test.
        rl_params['buffer_size'] = 10000

        logger.info(f"Test run timesteps: {rl_params['total_timesteps']}")
        logger.info(f"Test model save path: '{TEST_SAVE_DIR}'")

        # --- Instantiate and Run the RL Trainer ---
        logger.info("-" * 70)
        logger.info("Instantiating RLTrainer with the temporary test configuration...")
        trainer = RLTrainer(base_config_full=test_config)
        
        logger.info("Starting the RL training smoke test...")
        start_time = pd.Timestamp.now() if 'pandas' in sys.modules else None
        
        # The train method will run the full pipeline: env creation, callbacks, learning.
        final_model_path = trainer.train()
        
        if start_time:
            duration = pd.Timestamp.now() - start_time
            logger.info(f"Training smoke test completed in: {duration.total_seconds():.2f} seconds.")
        logger.info("-" * 70)

        # --- 3. Validate the Results ---
        logger.info("Performing results validation...")
        if not final_model_path:
            logger.error("************************************************************")
            logger.error("  VALIDATION FAILED: The trainer did not return a model path.")
            logger.error("************************************************************")
            return

        if not os.path.exists(final_model_path):
            logger.error("************************************************************")
            logger.error(f" VALIDATION FAILED: The final model file was not found at:")
            logger.error(f" '{final_model_path}'")
            logger.error("************************************************************")
            return

        logger.info(f"✅ SUCCESS: Final model file created at '{final_model_path}'.")
        
        logger.info("\n" + "=" * 70)
        logger.info("  RL TRAINING VALIDATION TEST PASSED: The RL stack is operational.")
        logger.info("=" * 70)

    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during the RL test: {e}", exc_info=True)
        logger.info("\n" + "=" * 70)
        logger.info("  RL TRAINING VALIDATION TEST FAILED due to a critical exception.")
        logger.info("=" * 70)

# --- 4. Script Entry Point ---
if __name__ == "__main__":
    # Import pandas here just for the timestamp functionality if needed
    try:
        import pandas as pd
    except ImportError:
        pass
        
    project_root_path = setup_environment()
    run_rl_training_test(project_root_path)
