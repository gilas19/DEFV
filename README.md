# DEFV - Debate-Enhanced Fact Verification

A research project exploring how multi-agent debate improves fact verification accuracy using fine-tuned T5 models on the FEVER dataset and CMV conversation data.

## Overview

This project implements a debate-enhanced fact verification system that uses two agents arguing for opposing stances (SUPPORTS vs REFUTES) to improve claim verification accuracy. The system compares baseline models against LoRA fine-tuned versions trained on Change My View (CMV) conversation data.

## Architecture

The system consists of three main components:

1. **Fine-tuning pipeline** (`src/fine_tune_model.py`): Trains T5 models on CMV debate data using LoRA
2. **Evaluation runner** (`src/evaluation_runner.py`): Runs experiments with different debate configurations
3. **Results aggregation** (`src/aggregate_results.py`): Analyzes results and generates visualizations

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Fine-tune Model (Optional)

Train a T5 model on CMV debate data:

```bash
python src/fine_tune_model.py --config t5_large.yaml
```

Configuration files are in `config/finetune/`. The system will use a pre-trained model if fine-tuning is skipped.

### 2. Run Experiments

Execute all experimental configurations:

```bash
python src/evaluation_runner.py
```

This generates results for:
- Baseline vs fine-tuned models
- No debate vs debate configurations  
- 1, 2, 4, 6 turns per agent
- Agent1 vs Agent2 as debate initiator

### 3. Analyze Results

Generate plots and analysis:

```bash
python src/aggregate_results.py
```

Results are saved to `analysis/plots/` including:
- Accuracy comparison plots
- Confusion matrices
- Correlation analysis
- Error pattern visualization

## Data

- **FEVER dataset**: Fact verification claims with gold labels (SUPPORTS/REFUTES/NOT ENOUGH INFO)
- **CMV data**: Change My View conversation trees for fine-tuning (`data/cmv_10_conversation_trees.json`)

## Contributors

- Shir Babian
- Yael Batat  
- Gilad Ticher
