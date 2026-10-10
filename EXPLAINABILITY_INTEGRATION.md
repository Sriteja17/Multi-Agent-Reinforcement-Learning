# Explainability Module — Install and Run

This bundle adds a **post-hoc explanation module** to the existing trained MAPPO decision trace. It does not retrain the agent or change its policy weights.

## 1. Files and locations

- `explainability.py` — explanation code.
- `install_explainability.py` — installer that copies the module into the right package and edits the existing trace file.

The installer expects the repository layout:

```text
Multi-Agent-Reinforcement-Learning/
├── explainability.py                 <- from this bundle
├── install_explainability.py         <- from this bundle
└── Marl/
    └── mappo/
        ├── trace.py                  <- existing project file; installer patches it
        └── ...
```

**Important:** this repository has `Marl/mappo/trace.py`. The installer detects and patches `trace.py` (and also supports `decision_trace.py` in other versions).

## 2. Install (WSL / Linux terminal)

Copy `explainability.py` and `install_explainability.py` from the downloaded ZIP into the repository root (the same folder that contains `Marl/` and `venv/`). Then run:

```bash
cd /mnt/c/Multi-Agent-Reinforcement-Learning
source venv/bin/activate
python install_explainability.py
```

The installer creates timestamped backups of files it changes. It also runs a Python syntax check. The syntax check does not replace testing with your actual checkpoint and installed dependencies.

If your repository is in a different directory, change the `cd` path to your own repository path.

## 3. Find your checkpoint

In the activated virtual environment, run:

```bash
find checkpoints -type f -name '*.pt'
```

Use one of the paths printed. If `checkpoints/use.pt` exists, the command in the next section should work as written.

## 4. Run a short test

```bash
python -m Marl.mappo.trace --checkpoint checkpoints/use.pt --steps 2 --seed 42
```

Replace `checkpoints/use.pt` with the actual checkpoint path if needed. Start with 2 steps because the counterfactual checks run the policy several extra times per agent decision. Once the short run succeeds, you can try `--steps 5`.

## 5. Where the results appear

The terminal prints an explanation for each agent's selected action. The JSON report is saved to:

```text
evaluation/explanations/decision_trace_seed42.json
```

The report includes the selected action and policy probability, top available alternatives, locally visible alerts, received structured messages, trust values, and tests that remove an incoming message or groups of alert features to measure how the action probability changes.

## 6. If it fails

- **`checkpoint not found`**: use the path returned by `find checkpoints -type f -name '*.pt'`.
- **`No module named Marl...`**: make sure the terminal is in the repository root and the virtual environment is activated. Run `pwd` and `ls` to confirm that you see the `Marl` folder.
- **Runtime/API error from the explainer**: the module was prepared from the repository's current source but has not been runtime-tested against your local environment/checkpoint. Keep the full traceback; the timestamped backup files let you restore the original trace if necessary.

## Interpretation note

This is post-hoc policy analysis, not a guaranteed causal explanation. If zeroing a message changes an action probability, it shows that the policy is sensitive to that input under this test; it does not prove that the message was the sole reason for the action.
