# run_optimization.py

"""
================================================================================
          Main Entry Point for Controller Optimization (DTAF v3.0)
================================================================================
This final script is the command-line interface to run automated optimization
for PID/FLC controllers, utilizing the robust, decoupled OptimizationManager.
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')

try:
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path: sys.path.insert(0, project_root)
    # The top-level script only needs to import the manager.
    from optimization_suite.optimization_manager import OptimizationManager
except (ImportError, ModuleNotFoundError) as e:
    print(f"\nFATAL ERROR: Could not set up Python path and import modules: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    """Parses arguments and runs the selected controller optimization."""
    parser = argparse.ArgumentParser(description="Run Automated Controller Optimization (PID/FLC).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--controller", required=True, choices=['PID', 'FLC'], help="Controller type to optimize.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_numeric)
    logger.info(f"Logging level set to: {args.log_level}")

    try:
        config_py_path = os.path.join(project_root, 'config', 'parameters.py')
        logger.info(f"Using configuration file: {config_py_path}")
        if not os.path.exists(config_py_path): raise FileNotFoundError(f"Configuration file not found: {config_py_path}")

        manager = OptimizationManager(config_path=config_py_path)
        # Note: No CLI overrides are passed here, but the **kwargs in the manager
        # makes it extensible for the future if needed.
        success = manager.run_optimization(controller_type=args.controller)

        if success:
            logger.info(f"✅ Optimization for {args.controller} completed successfully.")
            sys.exit(0)
        else:
            logger.error(f"❌ Optimization for {args.controller} failed. Check logs for details.")
            sys.exit(1)

    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
