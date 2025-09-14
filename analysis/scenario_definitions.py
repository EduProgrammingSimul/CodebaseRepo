# analysis/scenario_definitions.py

"""
================================================================================
    Adversarial Scenario Library & Drills (DTAF v6.0 - Expert Upgrade)
================================================================================
This version is expertly curated to include a new class of "Adversarial Drills"
and "Efficiency Probes" specifically designed to address the robustness and
efficiency deficiencies identified in the final analysis report. The goal is to
forge a controller that is not just high-performing, but also resilient and
operationally efficient.

Key Upgrades:
- **New Adversarial Drill**: 'cascading_grid_fault_and_recovery', a brutal,
  multi-stage scenario designed to break brittle policies by simulating a
  cascading failure. This directly targets the 'Robustness' deficiency.
- **New Efficiency Probe**: 'steady_state_efficiency_probe', a long-duration
  scenario flagged to use a reward scheme that heavily penalizes control effort,
  directly targeting the 'Control Efficiency' and 'Actuator Preservation'
  deficiencies.
- **Enhanced Scenario Parameters**: Existing scenarios like 'deceptive_sensor_noise'
  have been made more challenging to ensure the agent is hardened against a
  wider range of off-normal conditions.
================================================================================
"""

import numpy as np
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

# =============================================================================
# Load Profile & Event Helper Functions
# =============================================================================

def constant_load(load_mw: float) -> Callable[[float, int], float]:
    """Generates a constant load profile."""
    return lambda time_s, step: float(load_mw)

def gradual_load_change(initial: float, final: float, start_t: float, duration: float) -> Callable[[float, int], float]:
    """Generates a gradual, linear load ramp."""
    if duration <= 1e-6: duration = 1e-6
    def profile(time_s: float, step: int) -> float:
        if time_s < start_t:
            return float(initial)
        elif time_s < start_t + duration:
            fraction = (time_s - start_t) / duration
            return float(initial) + (float(final) - float(initial)) * fraction
        else:
            return float(final)
    return profile

def step_load_change(initial: float, final: float, step_t: float) -> Callable[[float, int], float]:
    """Generates an instantaneous step change in load."""
    return lambda time_s, step: float(final) if time_s >= float(step_t) else float(initial)

def multi_step_load_profile(steps: list) -> Callable[[float, int], float]:
    """Generates a profile with multiple sequential step changes."""
    def profile(time_s: float, step: int) -> float:
        current_load = steps[0][0]
        for load, start_time in steps:
            if time_s >= start_time:
                current_load = load
            else:
                break
        return float(current_load)
    return profile


# =============================================================================
# Main Scenario Definition Function (DTAF v6.0)
# =============================================================================

def get_scenarios(core_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Defines the complete, expert-curated library of all test and training scenarios.
    """
    scenarios: Dict[str, Dict[str, Any]] = {}
    try:
        sim_dt = core_config['simulation']['dt']
        initial_load_base = core_config.get('initial_conditions', {}).get('electrical_load_mw', 3008.5)
        eta_base = core_config.get('coupling', {}).get('eta_transfer', 0.98)
    except KeyError as e:
        logger.error(f"Failed to get required value from core_config: {e}", exc_info=True)
        return {}

    # === STANDARD VALIDATION SCENARIOS (The Basics) ===
    scenarios['baseline_steady_state'] = {
        'description': 'Baseline steady-state operation at 90% power.',
        'load_profile_func': constant_load(initial_load_base),
        'max_steps': int(500 / sim_dt),
        'reset_options': {'initial_power_level': 0.9}
    }
    scenarios['gradual_load_increase_10pct'] = {
        'description': 'Gradual load ramp from 90% to 100%.',
        'load_profile_func': gradual_load_change(initial_load_base * 0.9, initial_load_base, 20.0, 300.0),
        'max_steps': int(400 / sim_dt),
        'reset_options': {'initial_power_level': 0.9}
    }
    scenarios['sudden_load_increase_5pct'] = {
        'description': 'Sudden +5% load increase from nominal.',
        'load_profile_func': step_load_change(initial_load_base, initial_load_base * 1.05, 20.0),
        'max_steps': int(300 / sim_dt)
    }

    # === EFFICIENCY & ACTUATOR PRESERVATION PROBES (Targeted Drills) ===
    scenarios['steady_state_efficiency_probe'] = {
        'description': 'RL Drill: A long, quiet hold to enforce control efficiency.',
        'load_profile_func': constant_load(initial_load_base),
        'max_steps': int(1800 / sim_dt), # Very long duration
        'is_efficiency_probe': True # Custom flag for the reward function
    }

    # === ROBUSTNESS & ADVERSARIAL DRILLS (The Hardening Process) ===
    scenarios['deceptive_sensor_noise'] = {
        'description': 'Adversarial Test: High, dynamic sensor noise during a load ramp.',
        'load_profile_func': gradual_load_change(initial_load_base * 0.9, initial_load_base, 20.0, 300.0),
        'max_steps': int(400 / sim_dt),
        'reset_options': {'initial_power_level': 0.9},
        'adversarial_noise': {'active': True, 'initial_magnitude': 0.05, 'final_magnitude': 0.15, 'bias_magnitude': 8.0},
        'is_adversarial_drill': True # Flag for robustness reward
    }
    scenarios['parameter_randomization_drills'] = {
        'description': 'RL Drill: Train against randomized physics for generalization.',
        'load_profile_func': constant_load(initial_load_base),
        'max_steps': int(400 / sim_dt),
        'is_domain_randomization_drill': True,
        'is_adversarial_drill': True # Flag for robustness reward
    }
    
    # NEW ADVERSARIAL DRILL to directly target robustness deficiency
    scenarios['cascading_grid_fault_and_recovery'] = {
        'description': 'Adversarial Drill: A cascading grid fault followed by recovery demand.',
        'load_profile_func': multi_step_load_profile([
            (initial_load_base, 0.0),          # Start at nominal
            (initial_load_base * 0.8, 20.0),   # Sudden 20% load rejection (e.g., major line trip)
            (initial_load_base * 0.85, 120.0), # Grid stabilizes at a lower load
            (initial_load_base * 1.05, 150.0)  # Sudden demand to ramp up for recovery
        ]),
        'max_steps': int(500 / sim_dt),
        'env_modifications': [
            {'type': 'grid_power_imbalance', 'imbalance_mw': 50.0,
             'start_time': 20.0, 'end_time': 25.0} # Simulate instability during the fault
        ],
        'is_adversarial_drill': True # Flag for robustness reward
    }
    
    # === FINAL EXAM SCENARIO: The Ultimate Validation Test ===
    scenarios['combined_challenge_final_exam'] = {
        'description': 'Final Exam: Compound failure with grid fault, component degradation, and noise.',
        'load_profile_func': step_load_change(initial_load_base, initial_load_base * 1.1, 20.0),
        'max_steps': int(400 / sim_dt),
        'env_modifications': [
            {'type': 'parameter_ramp', 'parameter_path': ['coupling', 'eta_transfer'],
             'start_value': eta_base, 'end_value': eta_base * 0.90,
             'start_time': 50.0, 'duration': 150.0}
        ],
        'adversarial_noise': {'active': True, 'initial_magnitude': 0.02, 'final_magnitude': 0.05, 'bias_magnitude': 2.0},
        'is_adversarial_drill': True
    }
    
    # Final check to ensure all scenarios have a reset_options dictionary
    for name, config_dict in scenarios.items():
        if 'reset_options' not in config_dict:
            config_dict['reset_options'] = {}

    logger.info(f"Defined {len(scenarios)} total scenarios, including new adversarial drills and efficiency probes.")
    return scenarios
