"""
decision_trace.py

Human-readable decision trace of a TRAINED MAPPO checkpoint.

This is a read-only evaluation / visualisation tool. It does not change
training, rewards, environment dynamics or the existing evaluation
metrics. It runs ONE real rollout and prints, for every decision step
and for each of the 5 Blue agents:

    local observation summary
        -> messages received (generated at t-1) + directional trust
        -> action selected by the trained policy
    then, for the whole team:
        -> outgoing structured messages generated at t (usable at t+1)
        -> environment result (rewards)

Everything is reused from the project -- nothing is re-implemented:

    MAPPO, ppo.load()/ppo.eval()            mappo.py
    get_checkpoint_num_targets()            evaluate.py
    get_current_communication_for_agent()   evaluate.py  (t-1 messages + trust)
    select_action_greedy()                  evaluate.py  (same action path as eval)
    compute_padded_mask()                   action_mask.py (Adaptive Action Mask)
    describe_actions()                      action_mask.py (action type / target)
    pad_observation(), episode_is_done()    train.py
    ppo.get_outgoing_messages()             mappo.py (messages + 128-D vectors)
    ppo.get_trust_for_agent()               mappo.py (directional trust)
    CC4Env                                  env.py

Communication timing (identical to evaluate.py / train.py):

    step t : obs_t + messages_(t-1) + trust  ->  action_t
             obs_t                           ->  messages_t   (NOT seen at t)
    step t+1: messages_t become available

Trust
-----
Default: trust is FROZEN at the checkpoint's values, exactly like
Marl/mappo/evaluate.py. Pass --update-trust to additionally run the
project's own post-step trust update (train.evaluate_and_update_trust).
NOTE: that existing mechanism grades each message against CybORG
ground truth AFTER the step. The ground truth is never shown to the
agents; it only moves the trust values they see at the next step.

Usage
-----
    python -m Marl.mappo.decision_trace \\
        --checkpoint checkpoints/latest/mappo_final.pt \\
        --steps 5 --seed 42
"""

import argparse
import os

import time

import numpy as np

from .env import CC4Env
from .mappo import MAPPO
from .config import NUM_AGENTS, OBS_DIM, EPISODE_LENGTH
from .action_mask import compute_padded_mask, describe_actions
from .gnn_attention import NUM_HQ_SUBNETS, MAX_HOSTS
from .communication.schema import TargetType
from .train import pad_observation, episode_is_done, set_seed
from .evaluate import (
    RED_AGENTS,
    get_checkpoint_num_targets,
    get_current_communication_for_agent,
    select_action_greedy,
)

LINE = "=" * 60
THIN = "-" * 60


# ==========================================================
# Readable-name helpers (metadata only, never fed to agents)
# ==========================================================

def get_target_vocabularies(env):
    """
    Index -> name tables for message target_ids, using the same
    deterministic orderings documented in env.py:

        HOST   : sorted(state.hosts.keys())
        SUBNET : sorted(state.subnet_name_to_cidr.keys())

    Used only to print a target_id as a name.
    """
    state = env.cyborg.environment_controller.state
    hosts = sorted(state.hosts.keys())
    subnets = sorted(state.subnet_name_to_cidr.keys())
    return hosts, subnets


def target_to_str(message, hosts, subnets):
    if message.target_type == TargetType.HOST:
        if 0 <= message.target_id < len(hosts):
            return f"HOST:{hosts[message.target_id]}"
        return f"HOST:<id {message.target_id} outside current host list>"

    if message.target_type == TargetType.SUBNET:
        if 0 <= message.target_id < len(subnets):
            return f"SUBNET:{subnets[message.target_id]}"
        return f"SUBNET:<id {message.target_id} outside current subnet list>"

    return "NONE"


def format_message_lines(message, hosts, subnets, indent):
    """Seven schema fields, human readable."""
    pad = " " * indent
    return [
        f"{pad}Event: {message.event_type.name}",
        f"{pad}Target: {target_to_str(message, hosts, subnets)}",
        f"{pad}Threat: {message.threat_level.name}",
        f"{pad}Confidence: {message.confidence:.3f}",
        f"{pad}Status: {message.status.name}",
        f"{pad}Priority: {message.priority.name}",
    ]


def action_to_str(env, agent_name, action_index, meta):
    """
    '[14] Analyse operational_zone_b_subnet_user_host_8'
    Falls back to the env's own action label if no target is exposed.
    """
    action_type, target_host, source_zone = meta[action_index]

    if target_host is not None:
        return f"[{action_index}] {action_type} {target_host}"

    if source_zone is not None:
        label = env.action_labels(agent_name)[action_index]
        return f"[{action_index}] {action_type} (from zone {source_zone})  <{label}>"

    return f"[{action_index}] {action_type}"


# ==========================================================
# Printing
# ==========================================================

def print_observation_summary(env, name, real_dim, mask_real_valid):
    """
    Built ONLY from signals the agent itself legitimately has.

    env.get_host_alert_flags / get_zone_alert_flags re-read the same
    host-event log that BlueFlatWrapper projects into the agent's own
    malicious_processes / network_connections observation features
    (see env.py: "agent-legitimate signals"). They are NOT red-team
    ground truth, and are the same signals the Adaptive Action Mask uses.
    """
    host_flags = env.get_host_alert_flags(name)
    zone_flags = env.get_zone_alert_flags(name)

    flagged_hosts = sorted(h for h, f in host_flags.items() if f)
    flagged_zones = sorted(z for z, f in zone_flags.items() if f)

    print("  Local observation:")
    print(f"    Observation dimension: {OBS_DIM} (agent's real size {real_dim}, zero-padded)")
    print(f"    Flagged hosts: {', '.join(flagged_hosts) if flagged_hosts else 'none'}")
    print(f"    Flagged zones: {', '.join(flagged_zones) if flagged_zones else 'none'}")
    print(f"    Valid actions after Adaptive Action Mask: {mask_real_valid}")


def print_received(receiver_id, previous_messages, previous_structured,
                   trust_weights, hosts, subnets):
    if previous_messages is None:
        print("  Communication received: NONE (first decision step)")
        return

    print("  Communication received (generated at previous step):")

    for sender_id in range(NUM_AGENTS):
        if sender_id == receiver_id:
            continue

        # trust_weights = ppo.get_trust_for_agent(receiver_id)
        #               = trust_matrix[:, receiver_id]
        # -> trust_weights[sender] = how much THIS receiver trusts sender.
        trust = float(trust_weights[sender_id])
        message = previous_structured[sender_id]
        vec = previous_messages[sender_id]

        print()
        print(f"    Agent {sender_id}  (Agent {receiver_id}'s trust in Agent {sender_id} = {trust:.3f})")

        if message.is_empty():
            print("      Event: NONE  (empty message -- nothing reported)")
        else:
            for line in format_message_lines(message, hosts, subnets, 6):
                print(line)

        print(f"      -> encoded as {int(vec.numel())}-D vector "
              f"(L2 norm {float(vec.norm()):.3f}) before the receiver's trust weighting")
    print("")


# ==========================================================
# Main trace
# ==========================================================

def build_ppo(checkpoint_path):
    """Same construction order as evaluate.evaluate_checkpoint()."""
    num_host_targets, num_subnet_targets = get_checkpoint_num_targets(checkpoint_path)

    ppo = MAPPO(
        num_host_targets=num_host_targets,
        num_subnet_targets=num_subnet_targets,
    )
    ppo.load(checkpoint_path)
    ppo.eval()
    return ppo


def run_trace(checkpoint_path, steps, seed, red_agent_name, update_trust):
    ppo = build_ppo(checkpoint_path)

    env = CC4Env(red_agent_class=RED_AGENTS[red_agent_name])

    agent_names = sorted(env.possible_agents)
    if len(agent_names) != NUM_AGENTS:
        raise RuntimeError(
            f"Expected {NUM_AGENTS} Blue agents, got {len(agent_names)}"
        )

    obs_dims = env.get_observation_dims()

    set_seed(seed)
    obs_dict, info = env.reset(seed=seed)

    hosts, subnets = get_target_vocabularies(env)

    evaluator = None
    previous_info = info
    if update_trust:
        from .communication.evaluator import MessageEvaluator
        from .train import evaluate_and_update_trust
        evaluator = MessageEvaluator()

    steps = min(steps, EPISODE_LENGTH)

    print(LINE)
    print("TRAINED MAPPO DECISION TRACE")
    print(LINE)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Steps: {steps}")
    print(f"Seed: {seed}")
    print(f"Red agent: {RED_AGENTS[red_agent_name].__name__}")
    print("Action selection: greedy (same as evaluate.py default)")
    print("Trust: " + (
        "updated after each step via train.evaluate_and_update_trust"
        if update_trust else "frozen at checkpoint values (same as evaluate.py)"
    ))

    # Messages generated at t-1 (None at the first decision step).
    previous_vectors = None          # [NUM_AGENTS, 128] tensor
    previous_structured = None       # list[StructuredMessage], index = sender
    start = time.time()
    for t in range(steps):

        print()
        print(LINE)
        print(f"DECISION STEP {t}")
        print(LINE)

        # ---- padded observations (same as evaluate.py) ----
        obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)
        for i, name in enumerate(agent_names):
            obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

        host_active_masks = env.get_all_host_active_masks()
        assert host_active_masks.shape == (NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS)

        # ---- per-agent: observation, received comms, action ----
        actions_dict = {}

        for agent_id, name in enumerate(agent_names):

            print()
            print(f"Agent {agent_id}  ({name})")
            print(THIN)

            # Adaptive Action Mask (existing implementation)
            mask = compute_padded_mask(env, name)
            real_dim = obs_dims[name]
            n_valid = int(mask.sum())

            print_observation_summary(env, name, real_dim, n_valid)

            # Messages from t-1 + this receiver's trust (existing helper).
            received_messages, trust_weights = get_current_communication_for_agent(
                ppo=ppo,
                receiver_id=agent_id,
                previous_messages=previous_vectors,
            )

            print()
            print_received(
                agent_id, previous_vectors, previous_structured,
                trust_weights, hosts, subnets,
            )

            # Action from the trained policy (existing eval action path).
            action = select_action_greedy(
                ppo=ppo,
                observation=obs_array[agent_id],
                action_mask=mask,
                received_messages=received_messages,
                trust_weights=trust_weights,
                host_active_mask=host_active_masks[agent_id],
            )

            if not mask[action]:
                raise RuntimeError(
                    f"Agent {agent_id} selected action {action}, which the "
                    "existing action mask marks as invalid."
                )

            meta = describe_actions(env, name)
            print("  Selected action:")
            print(f"    {action_to_str(env, name, action, meta)}")

            actions_dict[name] = action

        # ---- outgoing communication from the CURRENT observation ----
        # Same call train.py makes (return_decoded=True). The structured
        # messages are built from the SAME sampled field_ids that were
        # encoded into the 128-D vectors, so what is printed is exactly
        # what is transmitted.
        (
            outgoing_vectors,
            outgoing_structured,
            _field_ids,
            _log_probs,
            _entropies,
        ) = ppo.get_outgoing_messages(
            obs_array,
            return_decoded=True,
            host_active_mask=host_active_masks,
        )
        outgoing_vectors = outgoing_vectors.detach()

        print()
        print(LINE)
        print("OUTGOING COMMUNICATION GENERATED AT THIS STEP")
        print("These messages become available at the NEXT decision step.")
        print(LINE)

        for sender_id in range(NUM_AGENTS):
            message = outgoing_structured[sender_id]
            print(f"Agent {sender_id}:")
            if message.is_empty():
                print("  Event: NONE  (empty message -- nothing to report)")
            else:
                for line in format_message_lines(message, hosts, subnets, 2):
                    print(line)
            print(f"  -> encoded as {int(outgoing_vectors[sender_id].numel())}-D vector")
            print()

        # ---- environment step ----
        (
            next_obs_dict,
            rewards_dict,
            terminated,
            truncated,
            info,
        ) = env.step(actions_dict)

        # ---- optional trust update (existing mechanism) ----
        if update_trust:
            evaluate_and_update_trust(
                ppo=ppo,
                evaluator=evaluator,
                env=env,
                outgoing_structured_messages=outgoing_structured,
                previous_info=previous_info,
                current_info=info,
            )
            previous_info = info

        

        print(LINE)
        print("ENVIRONMENT RESULT")
        print(LINE)
        for agent_id, name in enumerate(agent_names):
            print(f"Agent {agent_id} reward = {float(rewards_dict[name]):.3f}")

        if update_trust:
            print()
            print("Trust after update (row = receiver, column = sender):")
            matrix = ppo.get_trust_matrix()
            for receiver_id in range(NUM_AGENTS):
                cells = []
                for sender_id in range(NUM_AGENTS):
                    if sender_id == receiver_id:
                        cells.append("  -- ")
                    else:
                        cells.append(f"{float(matrix[sender_id, receiver_id]):.3f}")
                print(f"  Agent {receiver_id}: " + "  ".join(cells))

        # ---- one-step delay: only now do these become available ----
        previous_vectors = outgoing_vectors
        previous_structured = outgoing_structured
        obs_dict = next_obs_dict

        if episode_is_done(terminated, truncated):
            print()
            print("(episode ended)")
            break
    end = time.time()
    print(f"TOTAL infrence - {end - start}")

    print()
    print(LINE)
    print("DECISION TRACE COMPLETE")
    print(LINE)


def main():
    parser = argparse.ArgumentParser(
        description="Human-readable decision trace of a trained MAPPO checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained MAPPO .pt checkpoint")
    parser.add_argument("--steps", type=int, default=5, help="Number of decision steps to trace")
    parser.add_argument("--seed", type=int, default=42, help="Seed for env.reset() and torch/numpy")
    parser.add_argument("--red-agent", choices=sorted(RED_AGENTS.keys()), default="finite",
                        help="Red agent used in the rollout (default: finite)")
    parser.add_argument("--update-trust", action="store_true",
                        help="Also run the project's post-step trust update "
                             "(uses CybORG ground truth, never shown to agents). "
                             "Default: frozen trust, as in evaluate.py.")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")

    run_trace(
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        seed=args.seed,
        red_agent_name=args.red_agent,
        update_trust=args.update_trust,
    )


if __name__ == "__main__":
    main()
