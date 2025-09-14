# optimization_suite/flc_tuner.py

"""
================================================================================
          DEPRECATED FLC Tuner (DTAF v2.2)
================================================================================
This file is DEPRECATED and should NOT be used for Fuzzy Logic Controller (FLC)
tuning. Its functionality has been superseded by the more robust, multi-scenario
optimizer located in flc_optimizer.py.

This file is kept temporarily in the project with this notice to prevent
accidental use of outdated logic and to guide developers to the correct,
enhanced workflow. It can be safely deleted.
"""

import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# Define a clear deprecation message
DEPRECATION_MESSAGE = """
*******************************************************************************
* *
* WARNING: The 'flc_tuner.py' module is DEPRECATED.                         *
* *
* All FLC optimization should be performed using the robust, multi-scenario *
* optimizer located in:                                                     *
* *
* optimization_suite/flc_optimizer.py                                   *
* *
* Please update any scripts or workflows to call the functions within that  *
* module instead. This file will be removed in a future version.            *
* *
*******************************************************************************
"""

def _raise_deprecation_warning():
    """Logs and raises a warning to inform the user about the deprecation."""
    logger.error(DEPRECATION_MESSAGE)
    # Optionally, raise an exception to halt execution if this file is ever called
    # raise DeprecationWarning(DEPRECATION_MESSAGE)

# --- Calling the warning function at module level ---
# This ensures that a warning is logged as soon as this module is imported.
_raise_deprecation_warning()

# You can leave the rest of the file empty or include dummy functions that
# also raise the warning, to make it even more explicit.

def tune_flc_scaling(*args, **kwargs):
    """Dummy function for the deprecated tune_flc_scaling method."""
    _raise_deprecation_warning()
    print("DEPRECATED: Please use optimize_flc_scaling_de from flc_optimizer.py")
    return None

def save_flc_params(*args, **kwargs):
    """Dummy function for the deprecated save_flc_params method."""
    _raise_deprecation_warning()
    print("DEPRECATED: Please use the save functionality within flc_optimizer.py")
    return False