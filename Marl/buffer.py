"""
buffer.py

Multi-Agent Rollout Buffer for MAPPO.

Stores trajectories in the form

    [time_step][agent]

instead of flattening everything.

This matches how MAPPO is implemented in most
research codebases and makes centralized critics,
communication and trust modules easy to add later.
"""

import numpy as np
import torch

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    ROLLOUT_STEPS,
    DEVICE,
    GAMMA,
    GAE_LAMBDA,
)


class MAPPOBuffer:

    def __init__(self):

        self.clear()

    ###########################################################

    def clear(self):

        # ----------------------------------------------------
        # Local observations
        # shape = [T, N, OBS]
        # ----------------------------------------------------

        self.obs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, OBS_DIM),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Global observations
        # shape = [T, OBS*N]
        # ----------------------------------------------------

        self.global_obs = np.zeros(
            (
                ROLLOUT_STEPS,
                NUM_AGENTS * OBS_DIM,
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------

        self.actions = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.int64,
        )

        self.log_probs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        self.values = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        # Filled after rollout

        self.advantages = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        self.returns = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        self.action_masks = np.zeros(
            (
                ROLLOUT_STEPS,
                NUM_AGENTS,
                ACTION_DIM,
            ),
            dtype=bool,
        )

        self.ptr = 0

    ###########################################################

    def store(
        self,
        obs,
        global_obs,
        actions,
        log_probs,
        rewards,
        values,
        dones,
        action_masks,
    ):

        t = self.ptr

        self.obs[t] = obs
        self.global_obs[t] = global_obs

        self.actions[t] = actions
        self.log_probs[t] = log_probs

        self.rewards[t] = rewards
        self.values[t] = values
        self.action_masks[t] = action_masks
        self.dones[t] = dones

        self.ptr += 1

    ###########################################################

    def compute_advantages(
        self,
        last_values,
    ):
        """
        Compute GAE-Lambda advantages.

        last_values

        shape = [NUM_AGENTS]
        """

        gae = np.zeros(NUM_AGENTS, dtype=np.float32)

        for step in reversed(range(self.ptr)):

            if step == self.ptr - 1:

                next_values = last_values

            else:

                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + GAMMA
                * next_values
                * (1 - self.dones[step])
                - self.values[step]
            )

            gae = (
                delta
                + GAMMA
                * GAE_LAMBDA
                * (1 - self.dones[step])
                * gae
            )

            self.advantages[step] = gae

        self.returns = self.advantages + self.values

    ###########################################################

    def get_batches(self):
        """
        Convert rollout into tensors.

        Shapes

            obs
                [T,N,OBS]

            global_obs
                [T,N*OBS]

            actions
                [T,N]

            returns
                [T,N]

        """

        return {

            "obs":
                torch.tensor(
                    self.obs,
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "global_obs":
                torch.tensor(
                    self.global_obs,
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "actions":
                torch.tensor(
                    self.actions,
                    dtype=torch.long,
                    device=DEVICE,
                ),

            "log_probs":
                torch.tensor(
                    self.log_probs,
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "returns":
                torch.tensor(
                    self.returns,
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "advantages":
                torch.tensor(
                    self.advantages,
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "values":
                torch.tensor(
                    self.values,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
            "action_masks":
                torch.tensor(
                    self.action_masks,
                    dtype=torch.bool,
                    device=DEVICE,
                ),

        }

    ###########################################################

    def is_full(self):

        return self.ptr >= ROLLOUT_STEPS

    ###########################################################

    def __len__(self):

        return self.ptr