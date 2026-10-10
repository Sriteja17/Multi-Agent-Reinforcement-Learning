"""Post-hoc, analyst-facing explanations for the MAPPO cyber-defense policy.

Place this file at: Marl/mappo/explainability.py

The module does not modify the trained policy or use simulator ground truth.
It reports the model's action distribution, action-mask context, locally visible
alerts, received structured messages/trust, and input-ablation sensitivities.

Important interpretation:
- Attention weights are descriptive, not proof of causation.
- Counterfactual deltas measure model sensitivity to an input being removed;
  they do not prove that the input caused the real-world outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from .action_mask import describe_actions, explain_mask
from .config import ACTION_DIM, NUM_AGENTS
from .gnn_attention import (
    MAX_HOSTS,
    MISSION_DIM,
    NUM_HQ_SUBNETS,
    SUBNET_BLOCK_DIM,
    SUBNET_CONTEXT_DIM,
)
from .communication.schema import StructuredMessage, TargetType


def _json_safe(value: Any) -> Any:
    """Convert common NumPy / Torch values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def _to_1d_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().reshape(-1).tolist()
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def _policy_probs(
    ppo,
    observation,
    action_mask,
    received_messages=None,
    trust_weights=None,
    host_active_mask=None,
) -> torch.Tensor:
    """Run the actual actor and return masked action probabilities."""
    obs = ppo._to_tensor(observation, dtype=torch.float32)
    mask = ppo._to_tensor(action_mask, dtype=torch.bool)

    if obs.ndim != 1:
        raise ValueError(
            "explain_decision expects one agent observation with shape [OBS_DIM]."
        )

    prepared_host_mask = ppo._prepare_host_active_mask(host_active_mask, 1)

    if received_messages is not None:
        received_messages = ppo._to_tensor(received_messages, dtype=torch.float32)
    if trust_weights is not None:
        trust_weights = ppo._to_tensor(trust_weights, dtype=torch.float32)

    with torch.no_grad():
        logits = ppo.actor_forward(
            observations=obs,
            action_masks=mask,
            received_messages=received_messages,
            trust_weights=trust_weights,
            host_active_mask=prepared_host_mask,
        )
        return Categorical(logits=logits).probs.reshape(-1)


def _target_name(env, message: StructuredMessage) -> Optional[str]:
    """Resolve a message target ID for display only; does not alter policy inputs."""
    try:
        state = env.cyborg.environment_controller.state
        if message.target_type == TargetType.HOST:
            vocabulary = sorted(state.hosts.keys())
        elif message.target_type == TargetType.SUBNET:
            vocabulary = sorted(state.subnet_name_to_cidr.keys())
        else:
            return None
        if 0 <= int(message.target_id) < len(vocabulary):
            return str(vocabulary[int(message.target_id)])
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def _message_as_dict(env, message: Optional[StructuredMessage]) -> Optional[dict]:
    if message is None:
        return None
    result = message.as_dict()
    result["target_name"] = _target_name(env, message)
    return result


def _feature_ablation(
    ppo,
    observation,
    action_mask,
    received_messages,
    trust_weights,
    host_active_mask,
    selected_action: int,
    baseline_probability: float,
) -> Dict[str, dict]:
    """Measure selected-action probability after removing groups of alert bits."""
    obs = ppo._to_tensor(observation, dtype=torch.float32).detach().clone()
    results: Dict[str, dict] = {}

    # Each subnet block is: [subnet context | process alerts | connection alerts].
    process_indices = []
    connection_indices = []
    for slot in range(NUM_HQ_SUBNETS):
        base = MISSION_DIM + slot * SUBNET_BLOCK_DIM
        process_start = base + SUBNET_CONTEXT_DIM
        connection_start = process_start + MAX_HOSTS
        process_indices.extend(range(process_start, process_start + MAX_HOSTS))
        connection_indices.extend(range(connection_start, connection_start + MAX_HOSTS))

    groups = {
        "process_alert_features_removed": process_indices,
        "connection_alert_features_removed": connection_indices,
        "all_host_alert_features_removed": process_indices + connection_indices,
    }

    for label, indices in groups.items():
        if not indices or max(indices) >= obs.numel():
            results[label] = {"available": False, "reason": "Observation layout did not match expected CC4 dimensions."}
            continue
        perturbed = obs.clone()
        perturbed[indices] = 0.0
        probs = _policy_probs(
            ppo,
            perturbed,
            action_mask,
            received_messages=received_messages,
            trust_weights=trust_weights,
            host_active_mask=host_active_mask,
        )
        cf_probability = float(probs[selected_action].item())
        delta = baseline_probability - cf_probability
        results[label] = {
            "available": True,
            "selected_action_probability_after_ablation": cf_probability,
            "delta_probability_baseline_minus_ablation": delta,
            "interpretation": (
                "Removing this feature group increased the selected action's probability."
                if delta < -1e-6 else
                "Removing this feature group decreased the selected action's probability."
                if delta > 1e-6 else
                "Removing this feature group had negligible effect on the selected action's probability."
            ),
        }
    return results


def explain_decision(
    *,
    ppo,
    env,
    agent_id: int,
    agent_name: str,
    timestep: int,
    observation,
    action_mask,
    selected_action: int,
    received_messages=None,
    trust_weights=None,
    host_active_mask=None,
    previous_structured_messages: Optional[List[Optional[StructuredMessage]]] = None,
    top_k: int = 5,
    include_counterfactuals: bool = True,
) -> dict:
    """Create a structured explanation for one already-selected action.

    Call this after the normal action selection, supplying the same observation,
    action mask, incoming messages, trust vector, and host mask used for action
    selection. It is deliberately post-hoc and does not select a replacement action.
    """
    if not 0 <= int(agent_id) < NUM_AGENTS:
        raise ValueError(f"agent_id must be in [0, {NUM_AGENTS - 1}].")

    mask_np = np.asarray(action_mask, dtype=bool).reshape(-1)
    if mask_np.size != ACTION_DIM:
        raise ValueError(f"Expected action_mask length {ACTION_DIM}, got {mask_np.size}.")
    if not 0 <= int(selected_action) < mask_np.size:
        raise ValueError("selected_action is outside the action-mask range.")
    if not bool(mask_np[int(selected_action)]):
        raise ValueError("The selected action is marked invalid by the supplied action mask.")

    probs = _policy_probs(
        ppo,
        observation,
        action_mask,
        received_messages=received_messages,
        trust_weights=trust_weights,
        host_active_mask=host_active_mask,
    )
    selected_action = int(selected_action)
    selected_probability = float(probs[selected_action].item())
    prob_np = probs.detach().cpu().numpy()

    labels = list(env.action_labels(agent_name))
    meta = describe_actions(env, agent_name)

    def label_for(index: int) -> str:
        if 0 <= index < len(labels):
            return str(labels[index])
        return f"Action {index} (padded/unnamed action slot)"

    valid_indices = np.flatnonzero(mask_np)
    ordered_valid = sorted(valid_indices.tolist(), key=lambda i: prob_np[i], reverse=True)
    alternatives = [
        {"action_index": int(i), "label": label_for(int(i)), "probability": float(prob_np[i])}
        for i in ordered_valid[: max(1, int(top_k))]
    ]
    second_probability = float(prob_np[ordered_valid[1]]) if len(ordered_valid) > 1 else 0.0

    action_type, target_host, source_zone = (None, None, None)
    if 0 <= selected_action < len(meta):
        action_type, target_host, source_zone = meta[selected_action]

    host_flags = env.get_host_alert_flags(agent_name)
    zone_flags = env.get_zone_alert_flags(agent_name)
    flagged_hosts = sorted(str(host) for host, flagged in host_flags.items() if flagged)
    flagged_zones = sorted(str(zone) for zone, flagged in zone_flags.items() if flagged)

    local_evidence = {
        "flagged_hosts": flagged_hosts,
        "flagged_zones": flagged_zones,
        "selected_action_target_host": target_host,
        "selected_action_target_host_flagged": (
            bool(host_flags.get(target_host, False)) if target_host is not None else None
        ),
        "selected_action_source_zone": source_zone,
        "selected_action_source_zone_flagged": (
            bool(zone_flags.get(source_zone, False)) if source_zone is not None else None
        ),
    }

    if action_type in {"Restore", "Remove"} and target_host is not None:
        if host_flags.get(target_host, False):
            local_evidence["action_context_note"] = (
                f"The selected {action_type} action targets {target_host}, which currently has a visible process/connection alert."
            )
        else:
            local_evidence["action_context_note"] = (
                f"The selected {action_type} action targets {target_host}, but no current process/connection alert is reported for that host."
            )
    elif action_type == "BlockTrafficZone" and source_zone is not None:
        if zone_flags.get(source_zone, False):
            local_evidence["action_context_note"] = (
                f"The selected block action's source zone {source_zone} currently contains a flagged host."
            )
        else:
            local_evidence["action_context_note"] = (
                f"The selected block action's source zone {source_zone} has no currently flagged host."
            )
    elif action_type in {"Analyse", "Monitor", "DeployDecoy"}:
        local_evidence["action_context_note"] = (
            f"{action_type} is an investigative/monitoring-style action; the selected action is the policy's highest-probability available choice under the current inputs."
        )
    else:
        local_evidence["action_context_note"] = (
            "This is contextual evidence, not a claim that the alert alone caused the action."
        )

    # The existing helper reports adaptive suppressions. Only attach these as
    # actual constraints if the passed action mask matches the AAM mask.
    adaptive_mask, adaptive_reasons = explain_mask(env, agent_name)
    adaptive_padded = np.zeros(ACTION_DIM, dtype=bool)
    adaptive_padded[: min(ACTION_DIM, len(adaptive_mask))] = adaptive_mask[:ACTION_DIM]
    mask_is_adaptive = bool(np.array_equal(mask_np, adaptive_padded))
    suppressions = []
    if mask_is_adaptive:
        suppressions = [
            {"action_index": int(idx), "label": label_for(int(idx)), "reason": str(reason)}
            for idx, reason in adaptive_reasons.items()
        ]

    # Capture attention from the baseline forward pass. These weights are useful
    # context, but do not themselves establish that a message caused an action.
    attention_by_sender = {}
    # The actor does not refresh last_communication_attention when no messages
    # are supplied, so only read it on decisions that actually had messages;
    # otherwise it could contain weights from the previous agent's forward pass.
    attention = (
        getattr(ppo.actor, "last_communication_attention", None)
        if received_messages is not None else None
    )
    if attention is not None:
        attention_values = attention.detach().float().cpu().reshape(-1).tolist()
        for sender_id, value in enumerate(attention_values[:NUM_AGENTS]):
            if sender_id != agent_id:
                attention_by_sender[sender_id] = float(value)

    trust_values = _to_1d_list(trust_weights)
    message_values = None
    if received_messages is not None:
        message_values = ppo._to_tensor(received_messages, dtype=torch.float32).detach()
        if message_values.ndim == 3 and message_values.shape[0] == 1:
            message_values = message_values.squeeze(0)

    incoming_messages = []
    message_counterfactuals = []
    for sender_id in range(NUM_AGENTS):
        if sender_id == agent_id:
            continue

        structured = None
        if previous_structured_messages is not None and sender_id < len(previous_structured_messages):
            structured = previous_structured_messages[sender_id]

        trust = None
        if trust_values is not None and sender_id < len(trust_values):
            trust = float(trust_values[sender_id])

        incoming_messages.append({
            "sender_agent_id": sender_id,
            "trust_receiver_places_in_sender": trust,
            "communication_attention_weight": attention_by_sender.get(sender_id),
            "structured_message": _message_as_dict(env, structured),
            "message_available": message_values is not None,
        })

        if include_counterfactuals and message_values is not None:
            if message_values.ndim != 2 or message_values.shape[0] != NUM_AGENTS:
                raise ValueError(
                    "received_messages must have shape [NUM_AGENTS, COMMUNICATION_DIM] for message ablation."
                )
            cf_messages = message_values.clone()
            cf_messages[sender_id] = 0.0
            cf_probs = _policy_probs(
                ppo,
                observation,
                action_mask,
                received_messages=cf_messages,
                trust_weights=trust_weights,
                host_active_mask=host_active_mask,
            )
            cf_probability = float(cf_probs[selected_action].item())
            delta = selected_probability - cf_probability
            message_counterfactuals.append({
                "sender_agent_id": sender_id,
                "sender_structured_message": _message_as_dict(env, structured),
                "trust_receiver_places_in_sender": trust,
                "attention_weight": attention_by_sender.get(sender_id),
                "selected_action_probability_with_message": selected_probability,
                "selected_action_probability_when_sender_message_zeroed": cf_probability,
                "delta_probability_baseline_minus_ablation": delta,
                "interpretation": (
                    "Zeroing this sender's message reduced the selected action's probability; the policy is sensitive to this message in favor of the selected action."
                    if delta > 1e-6 else
                    "Zeroing this sender's message increased the selected action's probability; the policy is sensitive to this message against the selected action."
                    if delta < -1e-6 else
                    "Zeroing this sender's message had negligible effect on the selected action's probability."
                ),
            })

    feature_counterfactuals = {}
    if include_counterfactuals:
        feature_counterfactuals = _feature_ablation(
            ppo=ppo,
            observation=observation,
            action_mask=action_mask,
            received_messages=received_messages,
            trust_weights=trust_weights,
            host_active_mask=host_active_mask,
            selected_action=selected_action,
            baseline_probability=selected_probability,
        )

    # Sort by absolute effect size to surface the strongest sensitivities first.
    message_counterfactuals.sort(
        key=lambda item: abs(item["delta_probability_baseline_minus_ablation"]),
        reverse=True,
    )
    important_message_effects = [
        item for item in message_counterfactuals
        if abs(item["delta_probability_baseline_minus_ablation"]) >= 0.02
    ][:3]
    important_feature_effects = [
        {"feature_group": key, **value}
        for key, value in feature_counterfactuals.items()
        if value.get("available") and abs(value.get("delta_probability_baseline_minus_ablation", 0.0)) >= 0.02
    ]

    selected_label = label_for(selected_action)
    summary = (
        f"Agent {agent_id} selected '{selected_label}' with {selected_probability:.1%} policy probability "
        f"after applying the current action mask. The gap to the next-highest available action was "
        f"{selected_probability - second_probability:.1%}. This is a post-hoc explanation: alert and message "
        f"ablation results show model sensitivity, not guaranteed causal reasons."
    )

    return _json_safe({
        "schema_version": 1,
        "timestep": int(timestep),
        "agent_id": int(agent_id),
        "agent_name": str(agent_name),
        "selected_action": {
            "index": selected_action,
            "label": selected_label,
            "action_type": action_type,
            "target_host": target_host,
            "source_zone": source_zone,
            "policy_probability": selected_probability,
            "rank_among_available_actions": ordered_valid.index(selected_action) + 1,
            "probability_margin_over_second_choice": selected_probability - second_probability,
        },
        "top_available_actions": alternatives,
        "local_observation_evidence": local_evidence,
        "action_mask": {
            "mode": "adaptive_action_mask" if mask_is_adaptive else "structural_or_custom_mask",
            "valid_action_count": int(mask_np.sum()),
            "adaptive_suppression_reasons": suppressions,
        },
        "incoming_messages": incoming_messages,
        "message_counterfactuals": message_counterfactuals,
        "observation_feature_counterfactuals": feature_counterfactuals,
        "largest_message_sensitivities": important_message_effects,
        "largest_observation_sensitivities": important_feature_effects,
        "human_readable_summary": summary,
        "interpretation_limits": [
            "The actor is a neural policy; its internal reasoning is not directly available as a natural-language rationale.",
            "Attention weights are descriptive indicators and should not be treated as causal explanations.",
            "Counterfactual deltas are changes in the model's action probability when an input is removed, not proof of real-world causation.",
            "Local alerts are signals visible to the defender. Simulator ground truth is intentionally not used to justify the chosen action.",
        ],
    })


def format_explanation(report: dict) -> str:
    """Format a compact console explanation for an analyst."""
    lines = [
        "=" * 78,
        f"DECISION EXPLANATION | step={report['timestep']} | agent={report['agent_id']} ({report['agent_name']})",
        "=" * 78,
        report["human_readable_summary"],
        "",
        "Top available actions:",
    ]
    for item in report["top_available_actions"]:
        lines.append(f"  {item['probability']:7.2%}  [{item['action_index']}] {item['label']}")

    evidence = report["local_observation_evidence"]
    lines.extend([
        "",
        "Locally visible evidence:",
        f"  Flagged hosts: {', '.join(evidence['flagged_hosts']) if evidence['flagged_hosts'] else 'none'}",
        f"  Flagged zones: {', '.join(evidence['flagged_zones']) if evidence['flagged_zones'] else 'none'}",
        f"  Context: {evidence.get('action_context_note', 'not available')}",
    ])

    incoming = report["incoming_messages"]
    if incoming:
        lines.extend(["", "Incoming messages and model sensitivity:"])
        counterfactuals = {item["sender_agent_id"]: item for item in report["message_counterfactuals"]}
        for item in incoming:
            sender = item["sender_agent_id"]
            msg = item.get("structured_message")
            if msg is None:
                message_text = "structured message unavailable"
            else:
                target = msg.get("target_name") or msg.get("target_type")
                message_text = (
                    f"{msg.get('event_type')} / {target} / {msg.get('threat_level')} / "
                    f"status={msg.get('status')} / priority={msg.get('priority')}"
                )
            trust = item.get("trust_receiver_places_in_sender")
            attention = item.get("communication_attention_weight")
            extras = []
            if trust is not None:
                extras.append(f"trust={trust:.2f}")
            if attention is not None:
                extras.append(f"attention={attention:.2f}")
            cf = counterfactuals.get(sender)
            if cf is not None:
                delta = cf["delta_probability_baseline_minus_ablation"]
                extras.append(f"P(selected) change when removed={delta:+.1%}")
            lines.append(f"  Agent {sender}: {message_text}" + (f" ({', '.join(extras)})" if extras else ""))

    feature_effects = report["largest_observation_sensitivities"]
    if feature_effects:
        lines.extend(["", "Largest observation-feature sensitivities:"])
        for item in feature_effects:
            lines.append(
                f"  {item['feature_group']}: "
                f"P(selected) baseline-minus-ablation="
                f"{item['delta_probability_baseline_minus_ablation']:+.1%}"
            )

    lines.extend([
        "",
        "Caution: these are post-hoc model-sensitivity indicators, not causal proof.",
        "=" * 78,
    ])
    return "\n".join(lines)


def save_explanations_json(reports: List[dict], output_path: str | Path) -> Path:
    """Save one or more decision explanations as a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(reports), handle, indent=2, ensure_ascii=False)
    return path
