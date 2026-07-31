"""
train.py

Training script for MAPPO on CAGE Challenge 4 (CC4).

Wires together

    CC4Env       ->  environment
    MAPPOBuffer  ->  rollout storage
    MAPPO        ->  shared actor / centralized critic

This does NOT go through RLlib. It's a plain
rollout-collect -> GAE -> PPO-update loop, matching how
MAPPOBuffer and MAPPO.update() are implemented.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env import CC4Env
from .buffer import MAPPOBuffer
from .mappo import MAPPO

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    EPISODE_LENGTH,
    TOTAL_EPISODES,
    ROLLOUT_STEPS,
    SEED,
    PRINT_EVERY,
    SAVE_EVERY,
    CHECKPOINT_DIR,
    LOG_DIR,
)


############################################################
# Seeding
############################################################

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


############################################################
# Padding helpers
#
# Blue agents 0-3 use a 92-dim observation / 82-dim
# action space. Blue agent 4 uses 210 / 242.
#
# Everything is padded up to OBS_DIM / ACTION_DIM so the
# shared actor + centralized critic can take a fixed-size
# input. Padded action slots are hidden with the action
# mask so they're never sampled.
############################################################

def pad_observation(obs, real_dim):

    padded = np.zeros(OBS_DIM, dtype=np.float32)
    padded[:real_dim] = obs

    return padded


def build_action_mask(real_action_dim):

    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[:real_action_dim] = True

    return mask


############################################################
# Episode boundary helper
#
# EnterpriseMAE (RLlib-style MultiAgentEnv) usually returns
# "__all__" inside terminated / truncated. Fall back to
# all()-over-agents if it's missing.
############################################################

def episode_is_done(terminated, truncated):

    if "__all__" in terminated or "__all__" in truncated:

        return (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
        )

    return (
        all(terminated.values())
        or all(truncated.values())
    )


############################################################
# Main training loop
############################################################

def train():

    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ########################################################
    # Environment
    ########################################################

    env = CC4Env()

    agent_names = sorted(env.possible_agents)

    assert len(agent_names) == NUM_AGENTS, (
        f"Expected {NUM_AGENTS} blue agents, "
        f"found {len(agent_names)}: {agent_names}"
    )

    obs_dims = env.get_observation_dims()
    action_dims = env.get_action_dims()

    action_masks = {
        name: build_action_mask(action_dims[name])
        for name in agent_names
    }

    ########################################################
    # Agent / Buffer
    ########################################################

    ppo = MAPPO()
    buffer = MAPPOBuffer()

    ppo.eval()  # rollout collection -> no dropout/BN either way, kept for clarity

    ########################################################
    # Bookkeeping
    ########################################################

    obs_dict, info = env.reset(seed=SEED)

    episode_return = np.zeros(NUM_AGENTS, dtype=np.float32)
    episode_returns_log = []

    episode_count = 0
    update_count = 0

    ########################################################
    # Training History
    ########################################################

    episode_return_history = []

    actor_loss_history = []

    critic_loss_history = []

    entropy_history = []

    explained_variance_history = []

    approx_kl_history = []

    clip_fraction_history = []


    total_timesteps = TOTAL_EPISODES * EPISODE_LENGTH

    start_time = time.time()

    ########################################################
    # Rollout / Update Loop
    ########################################################

    for t in range(1, total_timesteps + 1):

        ####################################################
        # Build padded local + global observations
        ####################################################

        obs_array = np.zeros(
            (NUM_AGENTS, OBS_DIM),
            dtype=np.float32,
        )

        for i, name in enumerate(agent_names):

            obs_array[i] = pad_observation(
                obs_dict[name],
                obs_dims[name],
            )

        global_obs = obs_array.reshape(-1)

        ####################################################
        # Act
        ####################################################

        actions_dict = {}

        actions_arr = np.zeros(NUM_AGENTS, dtype=np.int64)
        log_probs_arr = np.zeros(NUM_AGENTS, dtype=np.float32)
        values_arr = np.zeros(NUM_AGENTS, dtype=np.float32)

        masks_arr = np.zeros(
            (NUM_AGENTS, ACTION_DIM),
            dtype=bool,
        )

        for i, name in enumerate(agent_names):

            mask = action_masks[name]
            masks_arr[i] = mask

            # action, log_prob, value, entropy = ppo.select_action(
            #     observation=obs_array[i],
            #     action_mask=mask,
            #     global_state=global_obs,
            # )

            action, log_prob, value, entropy = ppo.select_action(
                observation=obs_array[i],
                action_mask=mask,
                global_state=global_obs,
                agent_id=i,
            )

            actions_dict[name] = action

            actions_arr[i] = action
            log_probs_arr[i] = log_prob.item()
            values_arr[i] = (
                0.0 if value is None else value.item()
            )

        ####################################################
        # Step environment
        ####################################################

        next_obs_dict, rewards_dict, terminated, truncated, info = \
            env.step(actions_dict)

        rewards_arr = np.array(
            [rewards_dict[name] for name in agent_names],
            dtype=np.float32,
        )

        done_flag = episode_is_done(terminated, truncated)

        dones_arr = np.full(
            NUM_AGENTS,
            float(done_flag),
            dtype=np.float32,
        )

        ####################################################
        # Store transition
        ####################################################

        buffer.store(
            obs=obs_array,
            global_obs=global_obs,
            actions=actions_arr,
            log_probs=log_probs_arr,
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            action_masks=masks_arr,
        )

        episode_return += rewards_arr

        obs_dict = next_obs_dict

        ####################################################
        # Episode boundary
        ####################################################

        if done_flag:

            episode_count += 1
            # episode_returns_log.append(episode_return.sum())

            team_return = episode_return.sum()
            episode_returns_log.append(team_return)
            episode_return_history.append(team_return)

            if episode_count % PRINT_EVERY == 0:

                recent = episode_returns_log[-PRINT_EVERY:]
                mean_return = float(np.mean(recent))

                elapsed = time.time() - start_time

                print(
                    f"[episode {episode_count:6d}] "
                    f"team_return={mean_return:8.2f}  "
                    f"updates={update_count:5d}  "
                    f"elapsed={elapsed:7.1f}s"
                )

            if episode_count % SAVE_EVERY == 0:

                ckpt_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"mappo_ep{episode_count}.pt",
                )
                ppo.save(ckpt_path)

            episode_return[:] = 0.0

            obs_dict, info = env.reset()

        ####################################################
        # PPO Update
        ####################################################

        if buffer.is_full():

            last_obs_array = np.zeros(
                (NUM_AGENTS, OBS_DIM),
                dtype=np.float32,
            )

            for i, name in enumerate(agent_names):

                last_obs_array[i] = pad_observation(
                    obs_dict[name],
                    obs_dims[name],
                )

            last_global_obs = last_obs_array.reshape(-1)

            # last_value = ppo.get_value(last_global_obs)
            last_values = ppo.get_value(
                last_global_obs
            ).cpu().numpy()
            buffer.compute_advantages(last_values)

            ppo.train()
            stats = ppo.update(buffer)

            actor_loss_history.append(
                stats["actor_loss"]
            )

            critic_loss_history.append(
                stats["critic_loss"]
            )

            entropy_history.append(
                stats["entropy"]
            )

            # Add these once you implement them
            # explained_variance_history.append(stats["explained_variance"])
            # approx_kl_history.append(stats["approx_kl"])
            # clip_fraction_history.append(stats["clip_fraction"])

            ppo.eval()

            update_count += 1

            print(
                f"  -> update {update_count:5d}  "
                f"actor_loss={stats['actor_loss']:.4f}  "
                f"critic_loss={stats['critic_loss']:.4f}  "
                f"entropy={stats['entropy']:.4f}"
            )

            buffer.clear()

    ########################################################
    # Final checkpoint
    ########################################################

    final_path = os.path.join(CHECKPOINT_DIR, "mappo_final.pt")
    ppo.save(final_path)

    print(f"Training complete. Final checkpoint: {final_path}")

# Episode Return

    plt.figure(figsize=(10,5))
    plt.plot(episode_return_history)
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.savefig("logs/episode_return.png")

    ########################################################
    # Actor Loss
    ########################################################

    plt.figure(figsize=(10,5))
    plt.plot(actor_loss_history)
    plt.title("Actor Loss")
    plt.xlabel("PPO Update")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("logs/actor_loss.png")

    ########################################################
    # Critic Loss
    ########################################################

    plt.figure(figsize=(10,5))
    plt.plot(critic_loss_history)
    plt.title("Critic Loss")
    plt.xlabel("PPO Update")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("logs/critic_loss.png")

    ########################################################
    # Entropy
    ########################################################

    plt.figure(figsize=(10,5))
    plt.plot(entropy_history)
    plt.title("Entropy")
    plt.xlabel("PPO Update")
    plt.ylabel("Entropy")
    plt.grid()
    plt.savefig("logs/entropy.png")

    plt.show()


############################################################

if __name__ == "__main__":

    train()
