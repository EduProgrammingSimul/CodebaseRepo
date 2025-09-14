# PWR Controller Optimization & Monitoring Workbench (DTAF v3.0)

## 1. Project Overview

This project provides a comprehensive Python-based simulation framework for modeling the interaction between a Pressurized Water Reactor (PWR) nuclear power plant and the electrical grid. It focuses on the development, robust optimization, automated validation, and comparative analysis of advanced steam turbine governor control strategies.

The primary goal of this framework is to engineer and evaluate controllers (PID, FLC, RL) that are not just "optimal" under ideal conditions, but are robustly stable and reliable across a wide range of challenging operational and off-normal scenarios.

### Key Features

* **High-Fidelity Simulation Environment:** A modular `PWRGymEnvUnified` environment, compatible with the Gymnasium standard, that accurately models core physics, turbine dynamics, and grid interaction.
* **Advanced Controller Suite:** Implementations for PID, Fuzzy Logic (FLC), and a sophisticated Reinforcement Learning (RL) agent.
* **Robust Optimization Suite:** State-of-the-art optimizers for PID and FLC controllers that tune parameters against a full suite of validation scenarios to guarantee robustness.
* **Advanced RL Training:** A dedicated training pipeline for the RL agent, featuring a multi-stage curriculum to progressively increase task difficulty and ensure stable learning.
* **Automated End-to-End Validation & Reporting:** Automatic generation of comprehensive Markdown reports with advanced metrics and plots after any optimization or training run.
* **Interactive Streamlit UI:** A professional user interface for initiating controller optimization, training, analysis, and live simulation monitoring.

## 2. Project Architecture

The framework is designed with a clear separation of concerns, ensuring maintainability and robustness across its core components: `models`, `environment`, `controllers`, `optimization_suite`, `analysis`, and `ui`.

## 3. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <project-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements_ui.txt
    ```

## 4. How to Use the Workbench

The framework is designed to be run through clear, high-level commands. The recommended workflow is to first generate optimized controllers and then run a final comparative analysis.

### Step 1: Generate Your Controllers

It is highly recommended to first run the optimizers and trainers to generate stable parameters for your controllers.

* **To Tune the PID Controller:**
    This command runs the robust multi-scenario optimizer. The best, most stable gains will be saved to `config/optimized_controllers/PID_optimized.yaml`.
    ```bash
    python run_optimization.py --controller PID
    ```

* **To Tune the FLC Controller:**
    This tunes the FLC scaling factors for robust performance, saving the result to `config/optimized_controllers/FLC_optimized.yaml`.
    ```bash
    python run_optimization.py --controller FLC
    ```

* **To Train the RL Agent:**
    This will start the RL training process using the curriculum and fixed hyperparameters defined in `config/parameters.py`. The best model will be saved to the path specified in the configuration file.
    ```bash
    python run_training.py
    ```

### Step 2: Run a Full Comparative Analysis

After generating your controllers, you can run a comprehensive analysis to compare their performance across all scenarios.

This script will automatically discover all available controllers (both default and optimized versions) and generate a detailed report in the `results/reports/` directory.
```bash
python main_analysis.py
```

### Step 3: Use the Interactive UI (Optional)

For live monitoring and an interactive dashboard, launch the Streamlit application.
```bash
streamlit run ui/app.py
```

## 5. Core Configuration

All critical parameters are centralized in a single "source of truth" file for reliability and ease of modification:

* **`config/parameters.py`**: This file contains all core physics constants, default controller settings, safety limits, and the hyperparameters for the RL agent training process.
