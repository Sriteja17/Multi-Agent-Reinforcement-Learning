"""
evaluate.py

Evaluation script for a trained MAPPO model.

Usage:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/mappo_final.pt \
        --timesteps 1000

Optional:

    --red-agent random
    --red-agent finite
    --red-agent both

Default:
    --red-agent both

The --timesteps argument means the exact number of
environment transitions to execute.
"""

import argparse
import os
import time

import numpy as np
import torch

from .env import CC4Env
from .mappo import MAPPO

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    EPISODE_LENGTH,
)

from .train import (
    pad_observation,
    build_action_mask,
    episode_is_done,
)

from CybORG.Agents import (
    RandomSelectRedAgent,
    FiniteStateRedAgent,
)


############################################################
# Red Agents
############################################################

RED_AGENTS = {
    "random": RandomSelectRedAgent,
    "finite": FiniteStateRedAgent,
}


############################################################
# Load Trained Model
############################################################

def load_model(checkpoint_path):

    print()
    print("=" * 70)
    print("Loading trained MAPPO model")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    ppo = MAPPO()

    ppo.load(checkpoint_path)

    # Evaluation mode:
    # No dropout / training behaviour.
    ppo.eval()

    print("Model loaded successfully.")
    print("=" * 70)

    return ppo


############################################################
# Evaluate For Exact Number Of Timesteps
############################################################

@torch.no_grad()
def evaluate_timesteps(
    ppo,
    red_agent_class,
    num_timesteps,
    seed=100000,
):

    if num_timesteps <= 0:

        raise ValueError(
            "Number of timesteps must be greater than zero."
        )

    ########################################################
    # Environment
    ########################################################

    env = CC4Env(
        red_agent_class=red_agent_class
    )

    agent_names = sorted(
        env.possible_agents
    )

    if len(agent_names) != NUM_AGENTS:

        raise RuntimeError(
            f"Expected {NUM_AGENTS} agents, "
            f"but found {len(agent_names)}"
        )

    obs_dims = env.get_observation_dims()
    action_dims = env.get_action_dims()

    ########################################################
    # Initial Reset
    ########################################################

    obs_dict, info = env.reset(
        seed=seed
    )

    ########################################################
    # Statistics
    ########################################################

    total_return = 0.0

    agent_returns = np.zeros(
        NUM_AGENTS,
        dtype=np.float64,
    )

    episode_return = np.zeros(
        NUM_AGENTS,
        dtype=np.float64,
    )

    episode_returns = []

    timesteps_completed = 0
    episodes_completed = 0

    ########################################################
    # Start Timer
    ########################################################

    start_time = time.perf_counter()

    ########################################################
    # Evaluation Loop
    ########################################################

    while timesteps_completed < num_timesteps:

        ####################################################
        # Build Padded Observations
        ####################################################

        obs_array = np.zeros(
            (
                NUM_AGENTS,
                OBS_DIM,
            ),
            dtype=np.float32,
        )

        for i, name in enumerate(agent_names):

            obs_array[i] = pad_observation(
                obs_dict[name],
                obs_dims[name],
            )

        ####################################################
        # Global Observation
        ####################################################

        global_obs = obs_array.reshape(-1)

        ####################################################
        # Select Actions
        ####################################################

        actions_dict = {}

        for i, name in enumerate(agent_names):

            ################################################
            # Same action mask used during training
            ################################################

            mask = build_action_mask(
                action_dims[name]
            )

            ################################################
            # Greedy action
            #
            # The trained model is used directly.
            # No learning/update happens here.
            ################################################

            action, log_prob, value, entropy = (
                ppo.select_action(
                    observation=obs_array[i],
                    action_mask=mask,
                    global_state=global_obs,
                    agent_id=i,
                )
            )

            actions_dict[name] = action

        ####################################################
        # Environment Step
        ####################################################

        (
            next_obs_dict,
            rewards_dict,
            terminated,
            truncated,
            info,
        ) = env.step(
            actions_dict
        )

        ####################################################
        # Rewards
        ####################################################

        rewards_arr = np.array(
            [
                rewards_dict[name]
                for name in agent_names
            ],
            dtype=np.float64,
        )

        ####################################################
        # Accumulate Returns
        ####################################################

        agent_returns += rewards_arr

        episode_return += rewards_arr

        total_return += float(
            rewards_arr.sum()
        )

        timesteps_completed += 1

        ####################################################
        # Check Episode End
        ####################################################

        done = episode_is_done(
            terminated,
            truncated,
        )

        obs_dict = next_obs_dict

        ####################################################
        # Episode Finished
        ####################################################

        if done:

            episodes_completed += 1

            episode_team_return = float(
                episode_return.sum()
            )

            episode_returns.append(
                episode_team_return
            )

            episode_return[:] = 0.0

            ################################################
            # Start a fresh episode only if more timesteps
            # are still required.
            ################################################

            if timesteps_completed < num_timesteps:

                env = CC4Env(
                    red_agent_class=red_agent_class
                )

                agent_names = sorted(
                    env.possible_agents
                )

                obs_dims = (
                    env.get_observation_dims()
                )

                action_dims = (
                    env.get_action_dims()
                )

                obs_dict, info = env.reset(
                    seed=seed + episodes_completed
                )

    ########################################################
    # Stop Timer
    ########################################################

    total_time = (
        time.perf_counter()
        - start_time
    )

    ########################################################
    # Statistics
    ########################################################

    average_return_per_timestep = (
        total_return
        / timesteps_completed
    )

    timesteps_per_second = (
        timesteps_completed
        / total_time
    )

    time_per_timestep = (
        total_time
        / timesteps_completed
    )

    if len(episode_returns) > 0:

        average_episode_return = float(
            np.mean(
                episode_returns
            )
        )

    else:

        average_episode_return = 0.0

    ########################################################
    # Return Results
    ########################################################

    return {
        "timesteps": timesteps_completed,
        "total_return": total_return,
        "agent_returns": agent_returns,
        "episodes_completed": episodes_completed,
        "episode_returns": episode_returns,
        "average_episode_return": average_episode_return,
        "average_return_per_timestep": (
            average_return_per_timestep
        ),
        "total_time": total_time,
        "time_per_timestep": time_per_timestep,
        "timesteps_per_second": timesteps_per_second,
    }


############################################################
# Print Evaluation Results
############################################################

def print_results(
    red_name,
    results,
):

    print()
    print("=" * 70)
    print(
        f"Evaluation against: "
        f"{red_name}"
    )
    print("=" * 70)

    print(
        f"Timesteps executed        : "
        f"{results['timesteps']}"
    )

    print(
        f"Episodes completed       : "
        f"{results['episodes_completed']}"
    )

    print(
        f"Overall team return      : "
        f"{results['total_return']:.4f}"
    )

    print(
        f"Average episode return   : "
        f"{results['average_episode_return']:.4f}"
    )

    print(
        f"Average return/timestep  : "
        f"{results['average_return_per_timestep']:.6f}"
    )

    print()
    print("Return per Blue agent:")

    for i, value in enumerate(
        results["agent_returns"]
    ):

        print(
            f"  Agent {i}: "
            f"{value:.4f}"
        )

    print()
    print(
        f"Total time taken         : "
        f"{results['total_time']:.4f} seconds"
    )

    print(
        f"Time per timestep        : "
        f"{results['time_per_timestep']:.6f} seconds"
    )

    print(
        f"Timesteps per second     : "
        f"{results['timesteps_per_second']:.2f}"
    )

    print("=" * 70)


############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained MAPPO model for a "
            "specified number of timesteps."
        )
    )

    ########################################################
    # Checkpoint
    ########################################################

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to the trained MAPPO "
            "checkpoint (.pt)"
        ),
    )

    ########################################################
    # Number Of Timesteps
    ########################################################

    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help=(
            "Number of environment timesteps "
            "to evaluate."
        ),
    )

    ########################################################
    # Red Agent
    ########################################################

    parser.add_argument(
        "--red-agent",
        type=str,
        choices=[
            "random",
            "finite",
            "both",
        ],
        default="both",
        help=(
            "Red agent to evaluate against."
        ),
    )

    args = parser.parse_args()

    ########################################################
    # Validate
    ########################################################

    if args.timesteps <= 0:

        parser.error(
            "--timesteps must be greater than zero."
        )

    ########################################################
    # Load Model
    ########################################################

    ppo = load_model(
        args.checkpoint
    )

    ########################################################
    # Select Red Agents
    ########################################################

    if args.red_agent == "both":

        red_agents = [
            ("random", RandomSelectRedAgent),
            ("finite", FiniteStateRedAgent),
        ]

    else:

        red_agents = [
            (
                args.red_agent,
                RED_AGENTS[
                    args.red_agent
                ],
            )
        ]

    ########################################################
    # Header
    ########################################################

    print()
    print("=" * 70)
    print("MAPPO TIMESTEP EVALUATION")
    print("=" * 70)
    print(
        f"Requested timesteps : "
        f"{args.timesteps}"
    )
    print(
        f"Checkpoint           : "
        f"{args.checkpoint}"
    )
    print("=" * 70)

    ########################################################
    # Run Evaluation
    ########################################################

    all_results = {}

    for red_name, red_agent_class in red_agents:

        results = evaluate_timesteps(
            ppo=ppo,
            red_agent_class=red_agent_class,
            num_timesteps=args.timesteps,
        )

        all_results[
            red_name
        ] = results

        print_results(
            red_name,
            results,
        )

    ########################################################
    # Combined Result
    ########################################################

    if len(all_results) == 2:

        combined_return = (
            all_results["random"]["total_return"]
            + all_results["finite"]["total_return"]
        ) / 2.0

        print()
        print("=" * 70)
        print(
            "Combined average team return"
        )
        print("=" * 70)

        print(
            f"{combined_return:.4f}"
        )

        print("=" * 70)


############################################################

if __name__ == "__main__":

    main()