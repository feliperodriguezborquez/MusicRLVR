# MusicRLVR: Agentic Counterpoint Composer (GRPO Edition)

**Date**: January 13, 2026  
**Technique**: Group Relative Policy Optimization (GRPO) + Causal Transformer  
**Goal**: Solve strict Fuxian counterpoint problems and evolve to compose Bach-style polyphony.

---

## 🏗️ Architecture

The system uses a **Reinforcement Learning** approach tailored for reasoning and structure, inspired by **DeepSeek-R1**.

### 1. The Brain: `PolicyTransformer`
*   **Path**: `src/models/policy.py`
*   **Type**: Causal Transformer (Decoder-only).
*   **Function**: Receives a Cantus Firmus (melody) and generates a Counterpoint (harmony) token-by-token.
*   **Masking**: Uses strict **Causal Masking** to ensure autoregressive generation (no looking ahead).

### 2. The Engine: `GRPOEngine`
*   **Path**: `src/rl/grpo.py`
*   **Algorithm**: Group Relative Policy Optimization.
*   **Mechanism**:
    1.  Takes one input.
    2.  samples $G=4$ outputs.
    3.  Calculates Advantage based on how much better a sample is vs the group mean.
    4.  Updates weights to encourage the good samples.
    5.  **KL Penalty**: Keeps the model close to a "Reference Policy" to prevent mode collapse.

### 3. The Examiner: `CounterpointRules`
*   **Path**: `src/rl/rewards.py`
*   **Role**: Assigns numerical scores based on Music Theory (Fuxian Species 1).
    *   **+1** for Consonances.
    *   **-20** for Parallel 5ths/8ves (The "Cardinal Sin").

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch music21 tensorboard
```

### 2. Generate Data
Create 5,000 synthetic counterpoint exercises.
```bash
python generate_data.py
```
*(Optional) Download Bach data:* `python src/data/download_bach.py`

### 3. Train
Start the training loop (Validation enabled).
```bash
# Uncomment trainer.train() in the file first!
python train_mvp.py
```

### 4. Monitor
Watch the Agent learn via Tensorboard.
```bash
tensorboard --logdir logs
```

---

## 📂 Project Structure

```text
src/
├── data/
│   ├── dataset.py       # Pytorch Dataset & Tokenizer
│   └── download_bach.py # Music21 scraper
├── models/
│   └── policy.py        # The Actor (Transformer)
├── rl/
│   ├── grpo.py          # The Optimization Logic
│   └── rewards.py       # The Rules (Examiner)
└── utils/
    └── reproducibility.py
generate_data.py         # Synthetic Data Generator
train_mvp.py             # Main Entry Point
```

## 📈 2026 Audit Status
- **Reproducibility**: Seed set to 42.
- **Masking**: Causal Mask verified.
- **Eval**: 90/10 Train/Val split implemented.
- **Logging**: Tensorboard active.
