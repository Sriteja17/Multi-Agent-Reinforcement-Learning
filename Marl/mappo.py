"""
mappo.py

Part 1

MAPPO Agent

Contains

- Shared Actor
- Central Critic
- Action Selection
- Value Prediction
- Save / Load

The PPO update logic is implemented in Part 2.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from networks import MAPPOModel

from config import (
    DEVICE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
)


class MAPPO:

    ############################################################

    def __init__(self):

        self.device = DEVICE

        ########################################################
        # Networks
        ########################################################

        self.model = MAPPOModel().to(self.device)

        self.actor = self.model.actor
        self.critic = self.model.critic

        ########################################################
        # Optimizers
        ########################################################

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=LEARNING_RATE,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=LEARNING_RATE,
        )

        ########################################################

        self.max_grad_norm = MAX_GRAD_NORM

    ############################################################
    # Action Selection
    ############################################################

    @torch.no_grad()
    def select_action(
        self,
        observation,
        action_mask=None,
    ):
        """
        Parameters
        ----------
        observation

            shape

                [OBS_DIM]

        action_mask

            shape

                [ACTION_DIM]

            True  -> valid action

            False -> invalid action

        Returns
        -------

        action

        log_prob

        entropy
        """

        if not isinstance(observation, torch.Tensor):

            observation = torch.tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            )

        ########################################################

        logits = self.actor(observation)

        ########################################################
        # Action Masking
        ########################################################

        if action_mask is not None:

            if not isinstance(action_mask, torch.Tensor):

                action_mask = torch.tensor(
                    action_mask,
                    dtype=torch.bool,
                    device=self.device,
                )

            logits = logits.masked_fill(
                ~action_mask,
                -1e10,
            )

        ########################################################

        distribution = Categorical(logits=logits)

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        entropy = distribution.entropy()

        return (
            action.item(),
            log_prob,
            entropy,
        )

    ############################################################
    # Critic
    ############################################################

    @torch.no_grad()
    def get_value(
        self,
        global_state,
    ):
        """
        Parameters
        ----------

        global_state

            shape

                [1050]

        Returns

            scalar value estimate
        """

        if not isinstance(global_state, torch.Tensor):

            global_state = torch.tensor(
                global_state,
                dtype=torch.float32,
                device=self.device,
            )

        value = self.critic(global_state)

        return value.squeeze()

    ############################################################

    def train(self):

        self.model.train()

    ############################################################

    def eval(self):

        self.model.eval()

    ############################################################

    def save(
        self,
        path,
    ):
        """
        Save model checkpoint.
        """

        checkpoint = {

            "model": self.model.state_dict(),

            "actor_optimizer":
                self.actor_optimizer.state_dict(),

            "critic_optimizer":
                self.critic_optimizer.state_dict(),

        }

        torch.save(
            checkpoint,
            path,
        )

    ############################################################

    def load(
        self,
        path,
    ):
        """
        Load checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"],
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"],
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"],
        )

        ############################################################
    # Internal
    ############################################################

    def _apply_action_mask(
        self,
        logits,
        action_masks=None,
    ):
        """
        Apply action masks.

        logits:
            [B, ACTION_DIM]

        action_masks:
            [B, ACTION_DIM]
        """

        if action_masks is None:
            return logits

        return logits.masked_fill(
            ~action_masks.bool(),
            -1e10,
        )

    ############################################################
    # Actor Forward
    ############################################################

    def actor_forward(
        self,
        observations,
        action_masks=None,
    ):
        """
        observations

            Shape:
                [B, OBS_DIM]

        Returns

            logits
                [B, ACTION_DIM]
        """

        logits = self.actor(observations)

        logits = self._apply_action_mask(
            logits,
            action_masks,
        )

        return logits

    ############################################################
    # Critic Forward
    ############################################################

    def critic_forward(
        self,
        global_states,
    ):
        """
        global_states

            Shape:
                [B, NUM_AGENTS * OBS_DIM]

        Returns

            values
                [B]
        """

        values = self.critic(global_states)

        return values.squeeze(-1)

    ############################################################
    # Evaluate Actions
    ############################################################

    def evaluate_actions(
        self,
        observations,
        global_states,
        actions,
        action_masks=None,
    ):
        """
        Used during PPO updates.

        Returns

            log_probs
            entropy
            values
        """

        logits = self.actor_forward(
            observations,
            action_masks,
        )

        distribution = Categorical(
            logits=logits
        )

        log_probs = distribution.log_prob(
            actions
        )

        entropy = distribution.entropy()

        values = self.critic_forward(
            global_states
        )

        return (
            log_probs,
            entropy,
            values,
        )