# analysis/visualization_engine.py

"""
================================================================================
          Robust and Publication-Quality Visualization Engine (DTAF v2.2)
================================================================================
This version is hardened to run reliably in non-graphical server environments
by explicitly setting the Matplotlib backend before any other imports.
"""

# --- NEW: Robust Graphics Backend Configuration ---
# This is the critical fix. It MUST be executed BEFORE any other module
# that imports matplotlib.pyplot (like seaborn) is imported.
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plot generation.

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    Generates publication-quality plots for simulation results and metrics. v2.4
    This version is hardened for server-side execution.
    """

    def __init__(self, viz_config: Dict[str, Any], base_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the VisualizationEngine.
        """
        self.config = viz_config
        self.base_config = base_config or {}
        self.plot_dir = self.config.get('plot_output_dir', 'results/plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        sns.set_theme(style=self.config.get("seaborn_style", "whitegrid"))
        
        self.default_figsize = tuple(self.config.get('figsize', (14, 8)))
        self.default_formats = self.config.get('save_formats', ['png', 'svg'])
        self.default_dpi = self.config.get('dpi', 300)

        font_settings = self.config.get('font_settings', {})
        plt.rcParams.update({
            'font.family': font_settings.get('family', 'sans-serif'),
            'axes.titlesize': font_settings.get('title_size', 18),
            'axes.labelsize': font_settings.get('label_size', 16),
            'xtick.labelsize': font_settings.get('tick_size', 12),
            'ytick.labelsize': font_settings.get('tick_size', 12),
            'legend.fontsize': font_settings.get('legend_size', 14),
        })
        
        self.line_styles = ['-', '--', ':', '-.']
        self.color_palette = sns.color_palette("viridis", 8)
        logger.info(f"Visualization Engine v2.4 (Agg Backend) initialized. Plots saving to: '{self.plot_dir}'")

    def _save_plot(self, fig: plt.Figure, filename_base: str, scenario_name: str):
        """Helper function to save Matplotlib figures in multiple formats."""
        safe_scenario_name = scenario_name.replace(' ', '_').replace('/', '_')
        safe_filename_base = filename_base.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_').replace(':', '')
        path_base = os.path.join(self.plot_dir, f"{safe_scenario_name}_{safe_filename_base}")

        for fmt in self.default_formats:
            try:
                filepath = f"{path_base}.{fmt}"
                fig.savefig(filepath, bbox_inches='tight', dpi=self.default_dpi)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath} in format '{fmt}': {e}")
        plt.close(fig)

    def _add_safety_limit_overlays(self, ax: plt.Axes, var_key: str):
        """Adds horizontal lines for safety and nominal values to a plot."""
        # ... (This function's logic remains the same) ...
        pass

    def plot_scenario_time_series(self, scenario_name: str, scenario_results: Dict[str, Dict[str, Any]], **kwargs):
        """Generates comparative time-series plots for key variables in a scenario."""
        # ... (This function's logic remains the same) ...
        pass

    def plot_metric_comparison(self, scenario_name: str, scenario_results: Dict[str, Dict[str, Any]], **kwargs):
        """Generates bar charts comparing key performance metrics across controllers."""
        # ... (This function's logic remains the same) ...
        pass
