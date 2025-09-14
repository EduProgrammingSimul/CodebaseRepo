# analysis/config_loader.py

"""
================================================================================
          Robust Configuration Loader (DTAF v2.2)
================================================================================
This utility is responsible for loading the main project configuration from
the parameters.py file.

This enhanced version includes robust path handling and strict validation to
ensure that the simulation framework is always initialized with a valid and
complete configuration.
"""

import logging
import importlib.util
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define the default relative path to the configuration file
DEFAULT_CONFIG_PATH = os.path.join('config', 'parameters.py')

def load_config_from_py(project_root: str) -> Dict[str, Any]:
    """
    Loads the main configuration dictionary by executing the parameters.py file.
    This function is enhanced for robustness and provides clear error messages.

    Args:
        project_root (str): The absolute path to the project's root directory.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist at the expected path.
        AttributeError: If the config file is missing the 'get_config' function.
        TypeError: If the 'get_config' function does not return a dictionary.
        ImportError: If the file cannot be imported as a Python module.
    """
    config_filepath = os.path.join(project_root, DEFAULT_CONFIG_PATH)
    logger.info(f"Attempting to load configuration from: {config_filepath}")

    # 1. --- Strict Validation: Check if the file exists ---
    if not os.path.exists(config_filepath):
        logger.error(f"Configuration file not found at the absolute path: {config_filepath}")
        raise FileNotFoundError(f"Configuration file not found: {config_filepath}. Ensure the project root is correct and the file exists.")

    try:
        # 2. --- Dynamically Import the Configuration File ---
        # Use a unique module name to avoid conflicts in the Python environment
        module_name = f"pwr_project_config_{os.path.basename(config_filepath).replace('.py', '')}"

        spec = importlib.util.spec_from_file_location(module_name, config_filepath)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not create a module specification for the config file: {config_filepath}")

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module) # Execute the module to define its contents

        # 3. --- Strict Validation: Check for the get_config function ---
        if not hasattr(config_module, 'get_config') or not callable(config_module.get_config):
            logger.error(f"Config file '{config_filepath}' is missing a callable 'get_config' function.")
            raise AttributeError(f"The configuration file '{config_filepath}' must contain a 'get_config()' function.")

        # 4. --- Call the function and Validate the Output ---
        config = config_module.get_config()
        if not isinstance(config, dict):
             raise TypeError(f"The 'get_config()' function in '{config_filepath}' must return a dictionary, but returned type {type(config)}.")

        logger.info(f"Configuration loaded successfully from {config_filepath}")
        return config

    except ImportError as e:
        logger.error(f"Import error while loading config file '{config_filepath}': {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration from '{config_filepath}': {e}", exc_info=True)
        raise