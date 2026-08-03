# Multi-Agent Reinforcement Learning (MAPPO on CAGE Challenge 4)

This repository contains an implementation of Multi-Agent Proximal Policy Optimization (MAPPO) tailored for CAGE Challenge 4 (CC4). It leverages the CybORG environment to train autonomous cyber defense agents against varying adversary profiles.

## Table of Contents
- [Setup \& Installation](#setup--installation)
- [Exploring the Environment](#exploring-the-environment)
- [Training](#training)
- [Evaluation](#evaluation)

## Setup \& Installation

To get started, follow these steps to set up your environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anirudh110106/Multi-Agent-Reinforcement-Learning
   cd Multi-Agent-Reinforcement-Learning
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Install the CybORG package in editable mode:**
   ```bash
   pip install -e .
   ```

5. **Install system dependencies (required for some GUI/evaluation components):**
   ```bash
   sudo apt install python3-tk --assume-yes
   ```

## Exploring the Environment

The `playschool/` directory contains several useful scripts to help you understand the CC4 environment, agents, and action/observation spaces. Run these commands from the root directory of the repository:

- **Test Run (Random Agents):**
  Run a basic episode to see how the environment steps.
  ```bash
  python3 -m playschool.run_cc4
  ```

- **List Actions:**
  Inspect the available actions for the agents.
  ```bash
  python3 -m playschool.inspect_action
  ```

- **List Agent Info:**
  View information about the different agents in the environment.
  ```bash
  python3 -m playschool.inspect_agents
  ```

- **Inspect Observation Space:**
  Look at the raw observational space returned by the environment.
  ```bash
  python3 -m playschool.raw_observation
  ```

## Training

The MAPPO training implementation is located in the `Marl/` directory. The training process uses a curriculum schedule that progressively exposes the blue agents to more complex red agents (from `RandomSelectRedAgent` to `FiniteStateRedAgent`).

To start training:
```bash
python3 -m Marl.train
```

Checkpoints will be saved automatically in the `checkpoints/` directory. Training logs and plots (like episode return, actor/critic loss, and entropy) will be saved in the `evaluation/` directory.

## Evaluation

Once you have trained models (or want to evaluate an existing checkpoint), you can use the `evaluate.py` script. This script runs full episodes with a frozen policy and reports performance against different red agents.

- **Evaluate a single checkpoint:**
  ```bash
  python3 -m Marl.evaluate --checkpoint checkpoints/mappo_final.pt
  ```

- **Evaluate against a specific red agent (e.g., `finite`):**
  ```bash
  python3 -m Marl.evaluate --checkpoint checkpoints/mappo_final.pt --episodes 100 --red-agent finite
  ```

- **Evaluate a sweep of checkpoints:**
  ```bash
  python3 -m Marl.evaluate --sweep checkpoints/ --episodes 50
  ```

Available options for `--red-agent` are `random`, `finite`, or `both`.
