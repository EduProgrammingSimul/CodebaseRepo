# environment/__init__.py

"""
================================================================================
          Environment Package Initializer (DTAF v2.2)
================================================================================
This file makes the 'environment' directory a Python package, allowing for
clean and structured imports of the core simulation environment.

The __all__ variable explicitly defines the public API of this package.
"""

from .pwr_gym_env import PWRGymEnvUnified

# Explicitly declare the public API of the 'environment' package
__all__ = [
    'PWRGymEnvUnified'
]
