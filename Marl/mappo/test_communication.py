"""
test_communication.py  (manual input -> receiver model prediction)

MANUAL INPUT -> MODEL PREDICTION communication test.

You choose the sender, the receivers, and all seven StructuredMessage
fields by hand. The message is encoded with the project's real trained
MessageEncoder, and each receiver's ACTUAL trained SharedActor is run
forward on it, with the frozen trust from your checkpoint applied where
the trained architecture applies it.

Traced receiver-side path
-------------------------
There is NO component in this codebase that decodes a received message
back into a semantic judgement. MessageEvaluator is a training-time,
rule-based comparator used only for the trust UPDATE, so it is never
imported here. The real receiver path (gnn_attention.py SharedActor) is:

    local_hidden = self._get_local_hidden(observation)
    keys/values  = communication_key/value(messages)
    values       = values * trust.clamp(1e-4, 1.0)      <- frozen trust
    context, w   = communication_attention(query, keys, values)
    logits       = policy_head(concat([local_hidden, context]))

So "model-derived result" here means exactly:
    1. the attention weight the receiver assigns to this message slot
       (actor.last_communication_attention), and
    2. the shift in the receiver's action distribution versus a
       baseline pass with no message (greedy action change, top-k
       probabilities, KL divergence).

Nothing here modifies schema/encoder/decoder/trust/gnn_attention/mappo/
env/train/action_mask/evaluate. Trust is loaded and never updated; no
PPO update is performed.

Run from the project root:

    python -m Marl.mappo.test_communication
    python -m Marl.mappo.test_communication --checkpoint path/to/file.pt
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from .env import CC4Env
from .mappo import MAPPO
from .train import pad_observation, episode_is_done
from .action_mask import compute_padded_mask
from .config import NUM_AGENTS, OBS_DIM
from .communication.schema import (
    ConfidenceLevel,
    EventType,
    HostStatus,
    Priority,
    StructuredMessage,
    TargetType,
    ThreatLevel,
    confidence_level_to_value,
)


# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_PATH = (
    "/mnt/c/cyber/cage-challenge-4/"
    "checkpoints/latest/mappo_final.pt"
)

# Steps to advance the environment (random Blue actions) so receiver
# observations are not a trivial freshly-reset state.
WARMUP_STEPS = 5

TOP_K_ACTIONS = 5


# ============================================================================
# Checkpoint helpers
# ============================================================================

def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load the checkpoint ONCE and report exactly which file was read.

    weights_only=False is explicit: newer PyTorch defaults to True,
    which can reject checkpoints that store extra Python objects
    (optimizer state, trust state, value normalizer).
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    real_path = os.path.realpath(checkpoint_path)
    stat = os.stat(real_path)

    print()
    print("=" * 72)
    print("LOADING CHECKPOINT FILE")
    print("=" * 72)
    print(f"Requested : {checkpoint_path}")
    print(f"Resolved  : {real_path}")
    print(f"Size      : {stat.st_size:,} bytes")
    print(
        "Modified  : "
        + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    )

    try:
        checkpoint = torch.load(
            real_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        # Very old PyTorch without the weights_only argument.
        checkpoint = torch.load(real_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format. Expected a dictionary, "
            f"got {type(checkpoint).__name__}."
        )

    print(f"Top-level keys: {list(checkpoint.keys())}")

    if "model" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain a 'model' state_dict.")

    return checkpoint


def resolve_target_counts(
    checkpoint: dict,
) -> Tuple[int, Optional[int], str]:
    """
    Return (num_host_targets, num_subnet_targets, source).

    Preferred source: the explicit keys MAPPO.save() writes. If they are
    missing, fall back to the output size of the decoder's
    host_target_head / subnet_target_head in the saved state_dict. If
    neither exists, the checkpoint genuinely predates the split.
    """

    num_host_targets = checkpoint.get("num_host_targets")
    num_subnet_targets = checkpoint.get("num_subnet_targets")

    if num_host_targets is not None:
        source = "checkpoint keys"
    else:
        state = checkpoint["model"]

        host_keys = sorted(
            k for k in state
            if "host_target_head" in k and k.endswith("weight")
        )
        subnet_keys = sorted(
            k for k in state
            if "subnet_target_head" in k and k.endswith("weight")
        )

        if not host_keys:
            target_like = sorted(k for k in state if "target" in k)
            raise RuntimeError(
                "Checkpoint has no 'num_host_targets' key and no "
                "host_target_head weight, so it predates the host/subnet "
                "target split (or uses different names).\n"
                f"Top-level keys: {list(checkpoint.keys())}\n"
                "State-dict keys containing 'target':\n  "
                + "\n  ".join(target_like or ["<none>"])
            )

        num_host_targets = int(state[host_keys[0]].shape[0])
        num_subnet_targets = (
            int(state[subnet_keys[0]].shape[0]) if subnet_keys else None
        )
        source = f"inferred from {host_keys[0]}"

    num_host_targets = int(num_host_targets)

    if num_host_targets <= 0:
        raise RuntimeError(
            f"Invalid num_host_targets: {num_host_targets}"
        )

    if num_subnet_targets is not None:
        num_subnet_targets = int(num_subnet_targets)
        if num_subnet_targets <= 0:
            raise RuntimeError(
                f"Invalid num_subnet_targets: {num_subnet_targets}"
            )

    return num_host_targets, num_subnet_targets, source


def load_trained_mappo(checkpoint_path: str):
    """
    Build the same MAPPO architecture the checkpoint was trained with and
    load it. Trust is restored and left frozen.
    """

    checkpoint = load_checkpoint(checkpoint_path)

    num_host_targets, num_subnet_targets, source = resolve_target_counts(
        checkpoint
    )

    print(f"Host targets  : {num_host_targets}  ({source})")
    print(f"Subnet targets: {num_subnet_targets}")

    ppo = MAPPO(
        num_host_targets=num_host_targets,
        num_subnet_targets=num_subnet_targets,
    )

    model_state = checkpoint["model"]

    # Obsolete GNN buffers from older checkpoints; current gnn_attention.py
    # builds its own structural_adjacency / real_topology buffers.
    obsolete_keys = {"actor.gnn1.adjacency", "actor.gnn2.adjacency"}

    model_state = {
        key: value
        for key, value in model_state.items()
        if key not in obsolete_keys
    }

    missing, unexpected = ppo.model.load_state_dict(model_state, strict=False)

    expected_missing = {
        "actor.structural_adjacency",
        "actor.real_topology",
    }

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys after compatibility filtering:\n"
            + "\n".join(sorted(unexpected))
        )

    unexpected_missing = set(missing) - expected_missing

    if unexpected_missing:
        raise RuntimeError(
            "Unexpected missing model keys:\n"
            + "\n".join(sorted(unexpected_missing))
        )

    trust_state = checkpoint.get("trust_state")

    if trust_state is not None and hasattr(
        ppo.actor.communication, "load_trust_state"
    ):
        ppo.actor.communication.load_trust_state(trust_state)

    ppo.eval()

    print("Model         : loaded successfully")
    print("Trust         : loaded from checkpoint (frozen for this test)")
    print("=" * 72)

    return ppo, num_host_targets, num_subnet_targets


# ============================================================================
# Environment helpers (real receiver observations)
# ============================================================================

def build_observation_batch(agent_names, obs_dims, obs_dict) -> np.ndarray:
    """Pad every agent's raw CC4 observation, exactly as train.py does."""

    obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

    for i, name in enumerate(agent_names):
        obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

    return obs_array


def advance_environment(env, warmup_steps: int):
    """
    Reset, then step with random Blue actions so receiver observations are
    not a trivial freshly-reset state. Only the communication path is under
    test, so the specific actions do not matter.
    """

    obs_dict, info = env.reset()

    for _ in range(warmup_steps):

        actions = env.sample_actions()
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        if episode_is_done(terminated, truncated):
            obs_dict, info = env.reset()

    return obs_dict


# ============================================================================
# Input helpers
# ============================================================================

def choose_from_enum(enum_cls, title: str):
    """Interactively choose one IntEnum member."""

    members = list(enum_cls)

    print()
    print(title)

    for i, member in enumerate(members):
        print(f"  {i}. {member.name}")

    while True:
        raw = input("Select number: ").strip()

        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= index < len(members):
            return members[index]

        print(f"Choose a number from 0 to {len(members) - 1}.")


def choose_agent(num_agents: int, title: str) -> int:
    """Interactively choose one agent ID."""

    print()
    print(title)

    for agent_id in range(num_agents):
        print(f"  {agent_id}. Agent_{agent_id + 1}")

    while True:
        raw = input("Select number: ").strip()

        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= index < num_agents:
            return index

        print(f"Choose a number from 0 to {num_agents - 1}.")


def choose_receivers(sender: int, num_agents: int) -> List[int]:
    """Select receiving agents."""

    available = [a for a in range(num_agents) if a != sender]

    print()
    print("Receivers")
    print("  A. All other agents")

    for agent_id in available:
        print(f"  {agent_id + 1}. Agent_{agent_id + 1}")

    while True:
        raw = input(
            "Select 'A' for all, or comma-separated agent numbers: "
        ).strip().lower()

        if raw == "a":
            return available

        try:
            selected = [int(item.strip()) - 1 for item in raw.split(",")]
        except ValueError:
            print("Invalid selection.")
            continue

        if len(set(selected)) != len(selected):
            print("Do not select the same receiver twice.")
            continue

        invalid = [a for a in selected if a not in available]

        if invalid:
            print(
                "Invalid receiver(s): "
                + ", ".join(f"Agent_{a + 1}" for a in invalid)
            )
            continue

        return selected


def choose_target_id(num_targets: int) -> int:
    """Select a target ID within the given vocabulary."""

    print()
    print("Target ID")
    print(f"  Valid range: 0 - {num_targets - 1}")

    while True:
        raw = input("Enter target ID: ").strip()

        try:
            target_id = int(raw)
        except ValueError:
            print("Target ID must be an integer.")
            continue

        if 0 <= target_id < num_targets:
            return target_id

        print(f"Target ID must be between 0 and {num_targets - 1}.")


def build_manual_message(
    num_host_targets: int,
    num_subnet_targets: Optional[int],
) -> StructuredMessage:
    """Ask for all seven schema fields -- this IS the input."""

    print()
    print("=" * 72)
    print("MANUAL INPUT")
    print("=" * 72)

    event_type = choose_from_enum(EventType, "Event Type")
    target_type = choose_from_enum(TargetType, "Target Type")

    if target_type == TargetType.NONE:
        target_id = 0
    elif target_type == TargetType.SUBNET:
        if num_subnet_targets is None:
            raise RuntimeError(
                "This checkpoint has no SUBNET target vocabulary "
                "(num_subnet_targets is None), so a SUBNET target cannot "
                "be selected."
            )
        target_id = choose_target_id(num_subnet_targets)
    else:
        target_id = choose_target_id(num_host_targets)

    threat_level = choose_from_enum(ThreatLevel, "Threat Level")
    confidence_level = choose_from_enum(ConfidenceLevel, "Confidence")
    status = choose_from_enum(HostStatus, "Status")
    priority = choose_from_enum(Priority, "Priority")

    confidence = confidence_level_to_value(confidence_level)

    return StructuredMessage(
        event_type=event_type,
        target_type=target_type,
        target_id=target_id,
        threat_level=threat_level,
        confidence=confidence,
        status=status,
        priority=priority,
    )


# ============================================================================
# Real encoder integration (no reconstructed copy)
# ============================================================================

def message_to_field_ids(
    message: StructuredMessage,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert the manually chosen StructuredMessage into batch-of-1 ids."""

    levels = list(ConfidenceLevel)

    # Snap the numeric confidence back to the nearest discrete level index.
    confidence_id = min(
        range(len(levels)),
        key=lambda i: abs(
            confidence_level_to_value(levels[i]) - message.confidence
        ),
    )

    def ids(value: int) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.long, device=device)

    return {
        "event_type": ids(message.event_type),
        "target_type": ids(message.target_type),
        "target_id": ids(message.target_id),
        "threat_level": ids(message.threat_level),
        "confidence": ids(confidence_id),
        "status": ids(message.status),
        "priority": ids(message.priority),
    }


def encode_manual_message(
    ppo: MAPPO,
    message: StructuredMessage,
) -> torch.Tensor:
    """Encode using the project's REAL trained MessageEncoder."""

    encoder = ppo.actor.communication.encoder

    device = next(encoder.parameters()).device

    field_ids = message_to_field_ids(message, device)

    with torch.no_grad():
        vector = encoder.encode_from_ids(field_ids)

    return vector.squeeze(0)


# ============================================================================
# Real receiver-side forward pass
# ============================================================================

def run_receiver_model(
    ppo: MAPPO,
    env: CC4Env,
    agent_names,
    obs_array: np.ndarray,
    sender: int,
    receiver: int,
    encoded_vector: torch.Tensor,
):
    """
    Run the receiver's trained SharedActor once without the message and
    once with it, via ppo.actor_forward() (the function evaluate.py's
    select_action_greedy() uses).
    """

    device = next(ppo.actor.parameters()).device

    receiver_obs = torch.as_tensor(
        obs_array[receiver], dtype=torch.float32, device=device
    )

    mask = compute_padded_mask(env, agent_names[receiver])

    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)

    trust_row = ppo.get_trust_for_agent(receiver_id=receiver)  # frozen
    trust_score = float(trust_row[sender].item())

    messages_matrix = torch.zeros(
        NUM_AGENTS,
        encoded_vector.numel(),
        dtype=encoded_vector.dtype,
        device=device,
    )

    messages_matrix[sender] = encoded_vector.to(device)

    with torch.no_grad():

        baseline_logits = ppo.actor_forward(
            observations=receiver_obs,
            action_masks=mask_tensor,
            received_messages=None,
            trust_weights=None,
        )

        message_logits = ppo.actor_forward(
            observations=receiver_obs,
            action_masks=mask_tensor,
            received_messages=messages_matrix,
            trust_weights=trust_row,
        )

    # last_communication_attention is only populated by the WITH-message
    # call (the baseline call skips the attention branch).
    attention = getattr(ppo.actor, "last_communication_attention", None)

    attention_weight = None

    if attention is not None:
        flat = attention.reshape(-1, attention.shape[-1])
        attention_weight = float(flat[0, sender].item())

    baseline_dist = Categorical(logits=baseline_logits)
    message_dist = Categorical(logits=message_logits)

    baseline_action = int(torch.argmax(baseline_dist.probs).item())
    message_action = int(torch.argmax(message_dist.probs).item())

    top_probs, top_indices = torch.topk(
        message_dist.probs,
        k=min(TOP_K_ACTIONS, message_dist.probs.shape[-1]),
    )

    labels = env.action_labels(agent_names[receiver])

    top_actions = []
    for prob, idx in zip(top_probs.reshape(-1).tolist(),
                         top_indices.reshape(-1).tolist()):
        label = labels[idx] if idx < len(labels) else f"<pad_index_{idx}>"
        top_actions.append((label, prob))

    divergence = float(kl_divergence(message_dist, baseline_dist).item())

    return {
        "trust_score": trust_score,
        "attention_weight": attention_weight,
        "baseline_action": baseline_action,
        "message_action": message_action,
        "action_changed": baseline_action != message_action,
        "top_actions": top_actions,
        "kl_divergence": divergence,
    }


# ============================================================================
# Display
# ============================================================================

def format_target(message: StructuredMessage) -> str:

    if message.target_type == TargetType.HOST:
        return f"Host_id={message.target_id}"

    if message.target_type == TargetType.SUBNET:
        return f"Subnet_id={message.target_id}"

    return "None"


def print_manual_input(
    sender: int, receivers: List[int], message: StructuredMessage
) -> None:

    print()
    print("=" * 72)
    print("MANUAL INPUT")
    print("=" * 72)
    print(f"Sender:       Agent_{sender + 1}")
    print(
        "Receivers:    "
        + ", ".join(f"Agent_{r + 1}" for r in receivers)
    )
    print(f"Event:        {message.event_type.name}")
    print(f"Target Type:  {message.target_type.name}")
    print(f"Target:       {format_target(message)}")
    print(f"Threat Level: {message.threat_level.name}")
    print(f"Confidence:   {message.confidence:.3f}")
    print(f"Status:       {message.status.name}")
    print(f"Priority:     {message.priority.name}")


def print_encoded_vector(vector: torch.Tensor) -> None:

    print()
    print("=" * 72)
    print("ENCODED MESSAGE")
    print("=" * 72)
    print(f"Shape: {tuple(vector.shape)}")
    print(f"Norm:  {float(vector.norm()):.4f}")
    print(f"First 8 dims: {[round(v, 4) for v in vector[:8].tolist()]}")


def print_receiver_prediction(
    sender: int, receiver: int, result: dict
) -> None:

    attention = result["attention_weight"]
    attention_text = (
        f"{attention:.4f}  (trust already scales this)"
        if attention is not None
        else "unavailable (actor did not store attention weights)"
    )

    print()
    print(
        f"Agent_{receiver + 1} RECEIVER MODEL PREDICTION "
        f"(from Agent_{sender + 1})"
    )
    print("-" * 60)
    print(f"Frozen trust(sender->receiver):        {result['trust_score']:.3f}")
    print(f"Trained attention weight on this msg:  {attention_text}")
    print(f"Greedy action WITHOUT this message:    {result['baseline_action']}")
    print(f"Greedy action WITH this message:       {result['message_action']}")
    print(f"Action changed by this message:        {result['action_changed']}")
    print(f"KL(with_message || without_message):   {result['kl_divergence']:.5f}")
    print("Top action probabilities WITH this message:")
    for label, prob in result["top_actions"]:
        print(f"    {prob:.4f}  {label}")


# ============================================================================
# Main test
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT", CHECKPOINT_PATH),
        help="Path to the .pt checkpoint (default: CHECKPOINT_PATH).",
    )
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("CYBER MARL - MANUAL INPUT -> MODEL PREDICTION TEST")
    print("=" * 72)
    print("Message fields are YOUR input. Everything after encoding is")
    print("computed by the real trained model.")
    print("=" * 72)

    ppo, num_host_targets, num_subnet_targets = load_trained_mappo(
        args.checkpoint
    )

    print()
    print(f"Agents           : {NUM_AGENTS}")
    print(f"Host target IDs  : 0 - {num_host_targets - 1}")
    print(
        "Subnet target IDs: "
        + (
            f"0 - {num_subnet_targets - 1}"
            if num_subnet_targets is not None
            else "not available in this checkpoint"
        )
    )

    # ------------------------------------------------------------------
    # Real receiver observations
    # ------------------------------------------------------------------

    env = CC4Env()
    agent_names = sorted(env.possible_agents)

    assert len(agent_names) == NUM_AGENTS, (
        f"Expected {NUM_AGENTS} blue agents, found {len(agent_names)}: "
        f"{agent_names}"
    )

    obs_dims = env.get_observation_dims()

    print()
    print(
        f"Advancing the environment {WARMUP_STEPS} steps so receiver "
        f"observations aren't a trivial freshly-reset state..."
    )

    obs_dict = advance_environment(env, WARMUP_STEPS)
    obs_array = build_observation_batch(agent_names, obs_dims, obs_dict)

    # ------------------------------------------------------------------
    # Manual input
    # ------------------------------------------------------------------

    sender = choose_agent(NUM_AGENTS, "Sender")
    receivers = choose_receivers(sender, NUM_AGENTS)
    message = build_manual_message(num_host_targets, num_subnet_targets)

    print_manual_input(sender, receivers, message)

    # ------------------------------------------------------------------
    # Encode with the real trained encoder
    # ------------------------------------------------------------------

    encoded_vector = encode_manual_message(ppo, message)
    print_encoded_vector(encoded_vector)

    # ------------------------------------------------------------------
    # Real receiver-side model forward, per receiver
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("RECEIVER MODEL PREDICTION")
    print("=" * 72)

    results = {}

    for receiver in receivers:

        result = run_receiver_model(
            ppo=ppo,
            env=env,
            agent_names=agent_names,
            obs_array=obs_array,
            sender=sender,
            receiver=receiver,
            encoded_vector=encoded_vector,
        )

        results[receiver] = result

        print_receiver_prediction(sender, receiver, result)

    # ------------------------------------------------------------------
    # Frozen trust summary
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FROZEN TRUST FOR EACH RECEIVER")
    print("=" * 72)

    for receiver in receivers:

        trust_score = results[receiver]["trust_score"]

        if trust_score <= 0.25:
            level = "VERY LOW"
        elif trust_score <= 0.50:
            level = "LOW"
        elif trust_score <= 0.75:
            level = "HIGH"
        else:
            level = "VERY HIGH"

        print(
            f"Agent_{receiver + 1} trusts Agent_{sender + 1}: "
            f"{trust_score:.3f} ({level})"
        )

    print()
    print("=" * 72)
    print("TEST COMPLETE")
    print("=" * 72)
    print()
    print(
        "Note: trust was loaded from the checkpoint and never updated. "
        "No PPO update was performed. The message content above is "
        "exactly what you typed; everything under RECEIVER MODEL "
        "PREDICTION came from a real forward pass through the "
        "checkpoint's trained SharedActor."
    )


if __name__ == "__main__":
    main()