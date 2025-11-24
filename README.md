# Task-Dependent Evaluation of Reinforcement Learning Models in TAB and DAWH

A computational modeling project comparing two reinforcement learning models (Rescorla-Wagner and Stacked Probability) on rat behavioral data from two decision-making tasks: Two-Armed Bandit (TAB) and Dual Assignment with Hold (DAWH).

## Overview

This project implements a systematic computational modeling pipeline to:
- Compare two RL models (RW and SP) on empirical rat behavioral data
- Validate model recovery and parameter recovery capabilities
- Analyze model fits using likelihood-based and model-independent behavioral measures
- Investigate whether different tasks favor different cognitive strategies

## Models

### Rescorla-Wagner (RW) Model
Standard reinforcement learning model that updates action values based on prediction errors. Assumes animals learn expected values and choose accordingly.

**Parameters:**
- `α` (alpha): Learning rate (0 < α < 1)
- `β` (beta): Inverse temperature / choice determinism (β > 0)

### Stacked Probability (SP) Model
Extension of RW that multiplies the probability of reward on the ignored arm. Designed to better capture switching behavior in tasks with changing reward structures.

**Parameters:**
- `α` (alpha): Learning rate (0 < α < 1)
- `β` (beta): Inverse temperature / choice determinism (β > 0)

## Tasks

### Two-Armed Bandit (TAB)
Alternating blocks (~40 trials each) with constant reward probabilities per block. Simpler task structure favoring model-free learning.

### Dual Assignment with Hold (DAWH)
Reward probability of the unchosen arm increases each time it's ignored, creating an incentive to switch choices. More complex task structure that may favor model-based strategies.

## Dataset

- **Source**: Shin et al. (2021) and related studies
- **Total sessions**: 383 (215 TAB, 168 DAWH)
- **Data file**: `shin_data.pkl` (must be present in working directory)

## Project Structure

The analysis is organized into modular blocks that can be run sequentially in Google Colab or locally:

```
block1_setup.py              # Core models, fitting functions, and utilities
block2_parameter_recovery.py # Parameter recovery validation
block3_model_recovery.py     # Model recovery validation
block4_real_data.py          # Fit models to empirical data
block4.5_delta_ll.py         # Delta log-likelihood distribution analysis
block5_validation.py         # Model validation using behavioral measures
block5_runlength_analysis.py # Model-independent run-length analysis
block6_questions.py          # Answer psychological questions
```

## Quick Start

### Prerequisites

```python
pip install numpy pandas matplotlib scipy tqdm
```

### Running the Analysis

1. **Setup** (run first):
   ```python
   # Copy contents of block1_setup.py
   ```

2. **Parameter Recovery** (optional validation):
   ```python
   # Copy contents of block2_parameter_recovery.py
   ```

3. **Model Recovery** (optional validation):
   ```python
   # Copy contents of block3_model_recovery.py
   ```

4. **Real Data Analysis** (main analysis):
   ```python
   # Copy contents of block4_real_data.py
   # This creates the real_data_results DataFrame
   ```

5. **Additional Analyses**:
   ```python
   # Delta log-likelihood analysis
   # Copy contents of block4.5_delta_ll.py
   
   # Model validation
   # Copy contents of block5_validation.py
   
   # Run-length analysis
   # Copy contents of block5_runlength_analysis.py
   
   # Psychological questions
   # Copy contents of block6_questions.py
   ```

### For Google Colab

1. Upload `shin_data.pkl` to your Colab session
2. Copy each block into separate cells
3. Run cells sequentially
4. All figures and CSV results will be saved to the `figs/` folder

## Key Results

### Parameter Recovery
- Strong correlations between true and recovered parameters (r = 0.89-0.95)
- Validates that parameters can be reliably estimated from the data

### Model Recovery
- High accuracy in identifying data-generating models (91-100%)
- Confirms models are distinguishable for model comparison

### Model Comparison on Real Data
- **TAB task**: 114 sessions favor RW, 101 favor SP (53% vs 47%)
- **DAWH task**: 81 sessions favor RW, 87 favor SP (48% vs 52%)
- Chi-square test: χ² = 0.691, p = 0.406 (no significant task-dependent preference)

### Model-Independent Analysis
- Run-length analysis reveals qualitative differences: SP better captures switching behavior in DAWH task
- Highlights tension between likelihood-based and behavioral signature-based model selection

## Output Files

All outputs are saved to the `figs/` directory:

- `parameter_recovery.png` - Parameter recovery scatter plots
- `Model_Recovery_AIC_TAB.png` / `Model_Recovery_AIC_DAWH.png` - Model recovery confusion matrices
- `RW_alpha_by_task.png` / `RW_beta_by_task.png` - Parameter distributions for RW
- `SP_alpha_by_task.png` / `SP_beta_by_task.png` - Parameter distributions for SP
- `parameter_fitting_results.csv` - Full results from real data fitting
- `model_validation.png` - Behavioral measure correlations
- `delta_loglikelihood_analysis.png` - Delta log-likelihood distributions
- `run_length_analysis.png` - Model-independent run-length signatures

## Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `scipy` - Optimization and statistics
- `tqdm` - Progress bars

## Configuration

Key parameters can be adjusted in `block1_setup.py`:

```python
CFG = dict(
    n_trials_tab = 300,    # Number of trials per TAB session
    n_trials_dawh = 300,   # Number of trials per DAWH session
    n_datasets = 24,       # Number of datasets for recovery
    rw_restarts = 6,       # Optimization restarts for RW
    sp_restarts = 10,      # Optimization restarts for SP
)
```

## References

- Huh, N. et al. (2009). Model-based reinforcement learning under concurrent schedules of reinforcement in rodents. *Learning & Memory*, 16(5), 315-323.
- Shin, E.J. et al. (2021). Robust and distributed neural representation of action values. *eLife*, 10.

## Author

Aral Cay  
PSYC 51: Computational Models of Behavior  
November 2025

