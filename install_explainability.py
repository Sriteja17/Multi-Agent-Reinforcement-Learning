"""Install the post-hoc explainability module into this project's MAPPO trace.

Run this script from the repository root with the project virtualenv active:
    python install_explainability.py

It copies explainability.py into Marl/mappo/, patches the existing trace script
(trace.py or decision_trace.py), and backs up files before changing them.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> int:
    root = Path(__file__).resolve().parent
    source_module = root / "explainability.py"
    package_dir = root / "Marl" / "mappo"
    destination_module = package_dir / "explainability.py"
    # This repository currently uses trace.py. Support decision_trace.py too
    # for variants that renamed the file.
    trace_candidates = [package_dir / "trace.py", package_dir / "decision_trace.py"]
    trace_path = next((candidate for candidate in trace_candidates if candidate.is_file()), None)

    if not source_module.is_file():
        print(f"ERROR: {source_module} is missing. Keep this script next to explainability.py.")
        return 1
    if trace_path is None:
        print("ERROR: Could not find Marl/mappo/trace.py or Marl/mappo/decision_trace.py.")
        print("Run this script from the root of the Multi-Agent-Reinforcement-Learning repository.")
        return 1

    original = trace_path.read_text(encoding="utf-8")
    import_line = "from .explainability import explain_decision, format_explanation, save_explanations_json"

    if import_line in original:
        expected_markers = [
            "explanation_reports = []",
            "report = explain_decision(",
            "save_explanations_json(explanation_reports, output_path)",
        ]
        missing = [marker for marker in expected_markers if marker not in original]
        if missing:
            print(f"ERROR: {trace_path.name} appears partially patched. No files were changed.")
            print("Missing expected marker(s): " + ", ".join(missing))
            print("Restore the timestamped .bak file, then run the installer again.")
            return 1
        patched = original
        print(f"Explainability hooks already appear to be installed in {trace_path.relative_to(root)}.")
    else:
        anchors = {
            "import": 'LINE = "=" * 60',
            "reports_list": '    previous_structured = None       # list[StructuredMessage], index = sender\n    start = time.time()',
            "decision": '''            if not mask[action]:
                raise RuntimeError(
                    f"Agent {agent_id} selected action {action}, which the "
                    "existing action mask marks as invalid."
                )''',
            "save": "    end = time.time()",
        }
        for name, anchor in anchors.items():
            count = original.count(anchor)
            if count != 1:
                print(f"ERROR: Expected exactly one '{name}' insertion point in {trace_path.name}, found {count}.")
                print("No files were changed. Check that this is the expected repository version.")
                return 1

        patched = original.replace(
            anchors["import"],
            import_line + "\n\n" + anchors["import"],
            1,
        )
        patched = patched.replace(
            anchors["reports_list"],
            '    previous_structured = None       # list[StructuredMessage], index = sender\n'
            '    explanation_reports = []\n'
            '    start = time.time()',
            1,
        )
        decision_insert = anchors["decision"] + '''

            # Explain this already-selected action using the same inputs as the policy.
            report = explain_decision(
                ppo=ppo,
                env=env,
                agent_id=agent_id,
                agent_name=name,
                timestep=t,
                observation=obs_array[agent_id],
                action_mask=mask,
                selected_action=action,
                received_messages=received_messages,
                trust_weights=trust_weights,
                host_active_mask=host_active_masks[agent_id],
                previous_structured_messages=previous_structured,
                top_k=5,
                include_counterfactuals=True,
            )
            print(format_explanation(report))
            explanation_reports.append(report)'''
        patched = patched.replace(anchors["decision"], decision_insert, 1)
        save_insert = '''    output_path = f"evaluation/explanations/decision_trace_seed{seed}.json"
    saved_path = save_explanations_json(explanation_reports, output_path)
    print(f"Saved decision explanations to: {saved_path}")
    end = time.time()'''
        patched = patched.replace(anchors["save"], save_insert, 1)

    package_dir.mkdir(parents=True, exist_ok=True)

    # Back up any existing target files before replacing them.
    if destination_module.exists():
        module_backup = backup(destination_module)
        print(f"Backed up existing module to: {module_backup.relative_to(root)}")
    shutil.copy2(source_module, destination_module)

    if patched != original:
        trace_backup = backup(trace_path)
        trace_path.write_text(patched, encoding="utf-8")
        print(f"Backed up original decision trace to: {trace_backup.relative_to(root)}")
        print(f"Patched {trace_path.relative_to(root)}")

    # Syntax check only; this does not prove that the runtime integration succeeds.
    import subprocess
    check = subprocess.run(
        [sys.executable, "-m", "py_compile", str(destination_module), str(trace_path)],
        cwd=str(root),
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        print("ERROR: Python syntax check failed:")
        print(check.stdout)
        print(check.stderr)
        print(f"The backup of {trace_path.name} is available if you need to restore it.")
        return check.returncode

    print("\nInstalled the explainability module and passed the Python syntax check.")
    print("This is not yet a full runtime test; run a short trace to validate dependencies/checkpoint compatibility.")
    module_name = "Marl.mappo.trace" if trace_path.name == "trace.py" else "Marl.mappo.decision_trace"
    print(f"Next command: python -m {module_name} --checkpoint checkpoints/use.pt --steps 2 --seed 42")
    print("Use a different checkpoint path if checkpoints/use.pt does not exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
