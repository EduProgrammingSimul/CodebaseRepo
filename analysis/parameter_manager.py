# analysis/parameter_manager.py

"""
================================================================================
          Robust Parameter Manager (DTAF v2.2)
================================================================================
This module is responsible for loading, managing, and providing access to all
project configuration parameters from a Python configuration file.

This enhanced version includes strict validation, safe parameter access via
deep copies, and clear logging for CLI overrides to ensure simulation
integrity and reproducibility.
"""

import importlib.util
import os
import logging
import inspect
from typing import Any, Dict, Optional
import copy

logger = logging.getLogger(__name__)

class ParameterManager:
    """
    Manages loading and accessing configuration parameters for the simulation
    and analysis framework by importing a Python (.py) file as a module.
    """
    def __init__(self, config_filepath: str):
        """
        Initializes the ParameterManager by loading configuration from the
        specified Python module file.

        Args:
            config_filepath (str): The mandatory, absolute path to the configuration Python file.

        Raises:
            ValueError: If config_filepath is invalid or not a .py file.
            FileNotFoundError: If the specified config_filepath does not exist.
        """
        logger.info(f"Initializing ParameterManager with Python config file: {config_filepath}")

        # --- Strict Validation of config_filepath ---
        if not config_filepath or not isinstance(config_filepath, str):
            raise ValueError("Configuration filepath must be a non-empty string.")
        if not config_filepath.endswith('.py'):
             raise ValueError(f"Configuration filepath must point to a Python file (.py), but got: {config_filepath}")
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Configuration file not found at the specified path: {config_filepath}")

        self.config_path: str = config_filepath
        self.params: Dict[str, Any] = {}
        self._load_from_py_module()
        logger.info(f"Successfully loaded configuration from: {self.config_path}")

    def _load_from_py_module(self) -> None:
        """
        Loads parameters by importing the Python file and extracting uppercase variables.
        """
        logger.debug(f"Attempting to import Python module from: {self.config_path}")
        try:
            # Create a unique module name to avoid conflicts if loading multiple configs
            module_name = f"config_module_{os.path.basename(self.config_path).replace('.py', '')}"
            
            spec = importlib.util.spec_from_file_location(module_name, self.config_path)
            if spec is None or spec.loader is None:
                 raise ImportError(f"Could not create module spec for {self.config_path}")

            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            # Extract all uppercase variables as parameters, ignoring built-ins and modules
            loaded_params = {
                name: value for name, value in inspect.getmembers(config_module)
                if name.isupper() and not name.startswith('_') and not inspect.ismodule(value)
            }

            if not loaded_params:
                raise ValueError(f"No uppercase parameter variables found in config file: {self.config_path}. "
                                 "Ensure parameters are defined like 'MY_PARAM = value'.")

            self.params = loaded_params
            logger.debug(f"Parameters loaded: {list(self.params.keys())}")

        except Exception as e:
            logger.error(f"Failed to load or validate configuration from {self.config_path}: {e}", exc_info=True)
            raise

    def get_parameter(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a single top-level parameter value by its key.

        Args:
            key (str): The parameter key (e.g., 'CORE_PARAMETERS').
            default (Optional[Any]): The value to return if the key is not found.

        Returns:
            Any: The parameter value or the default.
        """
        return self.params.get(key, default)

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Returns the entire loaded configuration dictionary.

        The dictionary is returned as a deep copy to prevent any accidental
        modifications to the original "source of truth" configuration, which
        is crucial for simulation integrity.

        Returns:
            Dict[str, Any]: A deep copy of the dictionary containing all loaded parameters.
        """
        logger.debug("Providing a deep copy of all parameters.")
        return copy.deepcopy(self.params)

    def update_with_cli_args(self, cli_args_dict: Dict[str, Any]) -> None:
        """
        Updates the loaded parameters with values from CLI arguments.
        This method is designed to be robust and provide clear logging.
        Only non-None CLI arguments will override existing parameters.

        Args:
            cli_args_dict (Dict[str, Any]): A dictionary of parsed CLI arguments.
        """
        logger.info("Updating parameters with provided CLI arguments...")
        if 'CORE_PARAMETERS' not in self.params or 'rl_training_adv' not in self.params['CORE_PARAMETERS']:
            logger.warning("'CORE_PARAMETERS.rl_training_adv' path not found. CLI overrides for RL may not apply correctly.")
            return

        rl_params_target = self.params['CORE_PARAMETERS']['rl_training_adv']

        for cli_key, cli_value in cli_args_dict.items():
            if cli_value is not None:
                if cli_key in rl_params_target:
                    original_value = rl_params_target.get(cli_key)
                    if original_value != cli_value:
                        logger.info(f"  > CLI Override: ['rl_training_adv']['{cli_key}'] = {cli_value} (was: {original_value})")
                        rl_params_target[cli_key] = cli_value
                    else:
                        logger.debug(f"  > CLI Info: '{cli_key}' value from CLI matches existing config value.")
                else:
                    # This handles cases where a CLI arg might not have a direct key
                    # in the rl_training_adv dict, preventing unexpected additions.
                    logger.debug(f"  > CLI Info: Argument '{cli_key}' not found in 'rl_training_adv' params, skipping override.")
        
        logger.info("Finished updating parameters with CLI arguments.")