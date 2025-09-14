# analysis/__init__.py

"""
================================================================================
          Analysis Package Initializer (DTAF v3.0)
================================================================================
This file makes the 'analysis' directory a Python package, allowing for
clean and structured imports of its modules. This version has been corrected
to remove obsolete imports.
"""

from .config_loader import load_config_from_py
from .parameter_manager import ParameterManager
from .scenario_definitions import get_scenarios
from .scenario_executor import ScenarioExecutor
from .metrics_engine import MetricsEngine
from .visualization_engine import VisualizationEngine
from .report_generator import ReportGenerator

# Explicitly declare the public API of the 'analysis' package
# This removes the obsolete 'calculate_settling_time' function.
__all__ = [
    'load_config_from_py',
    'ParameterManager',
    'get_scenarios',
    'ScenarioExecutor',
    'MetricsEngine',
    'VisualizationEngine',
    'ReportGenerator'
]
