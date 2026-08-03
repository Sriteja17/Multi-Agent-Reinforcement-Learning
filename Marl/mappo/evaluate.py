"""
evaluate.py

Evaluate a saved MAPPO checkpoint against one or both CC4 red agents.

Runs full episodes with a frozen policy -- no gradient updates, no
buffer, no PPO update -- and reports per-red-agent statistics. This
is the counterpart to train.py's blended training-time logging: it
isolates performance against each red agent independently, instead
of a single team_return mean that mixes episode types together.

Place this file alongside train.py / mappo.py / buffer.py / env.py /
config.py in your package, then run it as a module (relative imports
require this):

    python -m yourpackage.evaluate --checkpoint checkpoints/mappo_ep2000.pt

    python -m yourpackage.evaluate --checkpoint checkpoints/mappo_ep2000.pt \
        --episodes 100 --red-agent finite

    python -m yourpackage.evaluate --sweep checkpoints/ --episodes 50

Replace "yourpackage" with your actual package name, or run the
underlying functions directly from a notebook / script inside the
package.
"""

import argparse
import glob
import os

import numpy as np
import torch

from .env import CC4Env
from .mappo import MAPPO

from .train import (
    pad_observation,
    build_action_mask,
    episode_is_done,
)

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    EPISODE_LENGTH,
)

from CybORG.Agents import (
    RandomSelectRedAgent,
    FiniteStateRedAgent,
)


RED_AGENTS = {
    "random": RandomSelectRedAgent,
    "finite": FiniteStateRedAgent,
}


############################################################
# Deterministic (greedy) action selection
#
# select_action() in mappo.py samples from the Categorical
# distribution -- correct for training exploration, wrong
# for evaluation. Eval should report the policy's best
# action, not a noisy sample of it. This reimplements just
# the actor forward pass + argmax, reusing ppo.actor and
# ppo.critic directly so mappo.py doesn't need editing.
############################################################

@torch.no_grad()
def select_action_greedy(ppo, observation, action_mask, global_state, agent_id):

    obs_t = torch.tensor(
        observation, dtype=torch.float32, device=ppo.device,
    )

    mask_t = torch.tensor(
        action_mask, dtype=torch.bool, device=ppo.device,
    )

    logits = ppo.actor(obs_t)
    logits = logits.masked_fill(~mask_t, -1e10)

    action = torch.argmax(logits).item()

    value = None

    if global_state is not None:

        global_t = torch.tensor(
            global_state, dtype=torch.float32, device=ppo.device,
        )

        value = ppo.critic(global_t)[agent_id]

        if ppo.value_norm is not None:
            value = ppo.value_norm.denormalize(value)

    return action, value


############################################################
# Run a fixed number of episodes against one red agent
############################################################

def run_episodes(
    ppo,
    red_agent_class,
    num_episodes,
    deterministic=True,
    base_seed=100_000,
):
    """
    base_seed is deliberately far outside the range used during
    training (SEED + episode_count) so eval episodes never overlap
    with episodes the policy actually trained on.
    """

    agent_names = None
    obs_dims = None
    action_dims = None
    action_masks = None

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):

        # Recreated every episode to mirror train.py's pattern --
        # if a plain env.reset() were sufficient, train.py wouldn't
        # rebuild CC4Env on every episode boundary either.
        env = CC4Env(red_agent_class=red_agent_class)

        if agent_names is None:

            agent_names = sorted(env.possible_agents)
            obs_dims = env.get_observation_dims()
            action_dims = env.get_action_dims()

            action_masks = {
                name: build_action_mask(action_dims[name])
                for name in agent_names
            }

        obs_dict, info = env.reset(seed=base_seed + ep)

        episode_return = np.zeros(NUM_AGENTS, dtype=np.float32)
        done = False
        t = 0

        while not done and t < EPISODE_LENGTH:

            obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

            for i, name in enumerate(agent_names):
                obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

            global_obs = obs_array.reshape(-1)

            actions_dict = {}

            for i, name in enumerate(agent_names):

                mask = action_masks[name]

                if deterministic:

                    action, _ = select_action_greedy(
                        ppo, obs_array[i], mask, global_obs, i,
                    )

                else:

                    action, _, _, _ = ppo.select_action(
                        observation=obs_array[i],
                        action_mask=mask,
                        global_state=global_obs,
                        agent_id=i,
                    )

                actions_dict[name] = action

            obs_dict, rewards_dict, terminated, truncated, info = \
                env.step(actions_dict)

            rewards_arr = np.array(
                [rewards_dict[name] for name in agent_names],
                dtype=np.float32,
            )

            episode_return += rewards_arr

            done = episode_is_done(terminated, truncated)
            t += 1

        episode_returns.append(float(episode_return.sum()))
        episode_lengths.append(t)

    return np.array(episode_returns), np.array(episode_lengths)


############################################################
# Evaluate one checkpoint against one or more red agents
############################################################

def evaluate_checkpoint(
    checkpoint_path,
    red_agent_names,
    num_episodes,
    deterministic,
):

    ppo = MAPPO()
    ppo.load(checkpoint_path)
    ppo.eval()

    results = {}

    for name in red_agent_names:

        red_agent_class = RED_AGENTS[name]

        returns, lengths = run_episodes(
            ppo,
            red_agent_class,
            num_episodes,
            deterministic=deterministic,
        )

        results[name] = {
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
            "mean_length": float(lengths.mean()),
            "n": num_episodes,
        }

        print(
            f"  {red_agent_class.__name__:24s} "
            f"mean={results[name]['mean']:8.2f}  "
            f"std={results[name]['std']:7.2f}  "
            f"min={results[name]['min']:8.2f}  "
            f"max={results[name]['max']:8.2f}  "
            f"avg_len={results[name]['mean_length']:5.1f}  "
            f"n={num_episodes}"
        )

    return results


############################################################
# CLI
############################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a single .pt checkpoint to evaluate.",
    )

    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Directory containing mappo_ep*.pt checkpoints to "
             "evaluate in episode order.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes per red agent.",
    )

    parser.add_argument(
        "--red-agent",
        type=str,
        choices=["random", "finite", "both"],
        default="both",
        help="Which red agent(s) to evaluate against.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of taking the greedy (argmax) "
             "action. Off by default -- eval should measure the "
             "policy's best behavior, not a noisy sample of it.",
    )

    args = parser.parse_args()

    if args.red_agent == "both":
        red_agent_names = ["random", "finite"]
    else:
        red_agent_names = [args.red_agent]

    deterministic = not args.stochastic

    if args.checkpoint:

        checkpoints = [args.checkpoint]

    elif args.sweep:

        checkpoints = sorted(
            glob.glob(os.path.join(args.sweep, "mappo_ep*.pt")),
            key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))),
        )

    else:

        parser.error("Provide either --checkpoint or --sweep.")
        return

    for ckpt in checkpoints:

        print(f"\n{os.path.basename(ckpt)}")
        evaluate_checkpoint(ckpt, red_agent_names, args.episodes, deterministic)


############################################################

if __name__ == "__main__":
    main()