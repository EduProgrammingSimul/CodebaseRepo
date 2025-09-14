# run_training.py

"""
================================================================================
          Main Entry Point for RL Agent Training (DTAF v4.1)
================================================================================
Version 4.1 hardens the logging system to use UTF-8 encoding for the file
handler and removes non-standard characters from log messages to guarantee
robust, cross-platform compatibility and prevent UnicodeEncodeError on
certain systems.
"""

import argparse
import logging
import os
import sys
import matplotlib
matplotlib.use('Agg')

# --- Centralized Logging Setup ---
def setup_logging(project_root: str, log_level: str = "INFO"):
    """
    Configures the root logger to write to both a file and the console.
    """
    log_filename = os.path.join(project_root, 'traininglog.log')
    log_level_numeric = getattr(logging, log_level.upper(), logging.INFO)
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)-30s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_numeric)

    # --- DEFINITIVE FIX 1: Use UTF-8 encoding for the file handler ---
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    root_logger.info(f"Logging configured. Output will be saved to '{log_filename}'")


def main():
    """Parses arguments and runs the RL agent training."""
    parser = argparse.ArgumentParser(
        description="Run Automated RL Agent Training using the DTAF framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for both console and file."
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override the total training timesteps from the config file."
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.dirname(__file__))
    setup_logging(project_root, args.log_level)

    try:
        from optimization_suite.optimization_manager import OptimizationManager

        config_py_path = os.path.join(project_root, 'config', 'parameters.py')
        logging.info(f"Using configuration file: {config_py_path}")
        if not os.path.exists(config_py_path):
            raise FileNotFoundError(f"Configuration file not found: {config_py_path}")

        manager = OptimizationManager(config_path=config_py_path)
        
        cli_overrides = {}
        if args.total_timesteps is not None:
            cli_overrides['total_timesteps'] = args.total_timesteps
        
        success = manager.run_optimization(controller_type='RL_AGENT', **cli_overrides)

        if success:
            # DEFINITIVE FIX 2: Use standard ASCII characters for log messages.
            logging.info(">> SUCCESS: RL Agent training process completed successfully.")
            sys.exit(0)
        else:
            logging.error(">> FAILURE: RL Agent training process failed. Check logs for details.")
            sys.exit(1)

    except Exception as e:
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
