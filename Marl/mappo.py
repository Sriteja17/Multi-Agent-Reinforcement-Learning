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
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .network import MAPPOModel
from .value_norm import ValueNorm


from .config import (
    DEVICE,
    LEARNING_RATE,
    ACTOR_LEARNING_RATE,
    USE_VALUE_NORM,
    CRITIC_LEARNING_RATE,
    MAX_GRAD_NORM,
    UPDATE_EPOCHS,
    MINIBATCH_SIZE,
    PPO_CLIP,
    VALUE_LOSS_COEF,
    ENTROPY_COEF,
    NORMALIZE_ADVANTAGES,
    NUM_AGENTS,
    OBS_DIM,
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
            lr=ACTOR_LEARNING_RATE,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LEARNING_RATE,
        )

        ########################################################

        self.max_grad_norm = MAX_GRAD_NORM
        # Value Normalization
        self.value_norm = None
        if USE_VALUE_NORM:
            self.value_norm = ValueNorm(
                device=self.device,
            )

    ############################################################
    # Action Selection
    ############################################################

    @torch.no_grad()
    # def select_action(
    #     self,
    #     observation,
    #     action_mask=None,
    #     global_state = None,
    # ):
    #     """
    #     Parameters
    #     ----------
    #     observation

    #         shape

    #             [OBS_DIM]

    #     action_mask

    #         shape

    #             [ACTION_DIM]

    #         True  -> valid action

    #         False -> invalid action

    #     Returns
    #     -------

    #     action

    #     log_prob

    #     entropy
    #     """

    #     if not isinstance(observation, torch.Tensor):

    #         observation = torch.tensor(
    #             observation,
    #             dtype=torch.float32,
    #             device=self.device,
    #         )

    #     ########################################################

    #     logits = self.actor(observation)

    #     ########################################################
    #     # Action Masking
    #     ########################################################

    #     if action_mask is not None:

    #         if not isinstance(action_mask, torch.Tensor):

    #             action_mask = torch.tensor(
    #                 action_mask,
    #                 dtype=torch.bool,
    #                 device=self.device,
    #             )

    #         logits = logits.masked_fill(
    #             ~action_mask,
    #             -1e10,
    #         )

    #     ########################################################

    #     distribution = Categorical(logits=logits)

    #     action = distribution.sample()

    #     log_prob = distribution.log_prob(action)

    #     entropy = distribution.entropy()

    #     return (
    #         action.item(),
    #         log_prob,
    #         entropy,
    #     )

    @torch.no_grad()
    def select_action(
        self,
        observation,
        action_mask=None,
        global_state=None,
        agent_id = None,
    ):
        """
        Select an action using the shared actor.

        Parameters
        ----------
        observation : array-like
            Local observation of one blue agent.

        action_mask : array-like, optional
            Boolean mask of valid actions.

        global_state : array-like, optional
            Concatenated observations of all agents.
            Used by the centralized critic.

        Returns
        -------
        action : int
            Sampled action.

        log_prob : torch.Tensor
            Log probability of sampled action.

        value : torch.Tensor or None
            Critic value estimate.

        entropy : torch.Tensor
            Policy entropy.
        """

        ########################################################
        # Observation
        ########################################################

        if not isinstance(observation, torch.Tensor):

            observation = torch.tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            )

        ########################################################
        # Actor Forward
        ########################################################

        logits = self.actor(observation)

        ########################################################
        # Action Mask
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
        # Distribution
        ########################################################

        distribution = Categorical(logits=logits)

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        entropy = distribution.entropy()

        ########################################################
        # Centralized Critic
        ########################################################

        value = None

        if global_state is not None:

            if not isinstance(global_state, torch.Tensor):

                global_state = torch.tensor(
                    global_state,
                    dtype=torch.float32,
                    device=self.device,
                )

            # value = self.critic(global_state).squeeze()
            value = self.critic(global_state)[agent_id]
            if self.value_norm is not None:
                value = self.value_norm.denormalize(
                    value,
                )
        ########################################################

        return (
            action.item(),
            log_prob.detach(),
            None if value is None else value.detach(),
            entropy.detach(),
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

        # value = self.critic(global_state)

        # return value.squeeze()
        # return self.critic(global_state)

        value = self.critic(global_state)
        if self.value_norm is not None:
            value = self.value_norm.denormalize(
                value,
            )
        return value

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

                "value_norm":
                    None if self.value_norm is None
                    else self.value_norm.state_dict(),
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
        if (self.value_norm is not None and checkpoint["value_norm"] is not None):
            self.value_norm.load_state_dict(checkpoint["value_norm"])
            
    # Internal
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

        # return values.squeeze(-1)
        return self.critic(global_states)

    ############################################################
    # Evaluate Actions
    ############################################################

    def evaluate_actions(
        self,
        observations,
        global_states,
        actions,
        agent_ids,
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

        # values = self.critic_forward(
        #     global_states
        # )
        
        values = self.critic_forward(global_states)

        values = values[
            torch.arange(
                values.size(0),
                device=self.device,
            ),
            agent_ids,
        ]



        return (
            log_probs,
            entropy,
            values,
        )
# update stuff:-

    def update(
        self,
        buffer,
    ):
        """
        Perform one MAPPO update.

        Part 3A

        - Prepare tensors
        - Normalize advantages
        - Create minibatches
        - Compute PPO losses

        Part 3B will perform the optimizer steps.
        """

        batch = buffer.get_batches()

        obs = batch["obs"]
        global_obs = batch["global_obs"]

        actions = batch["actions"]

        old_log_probs = batch["log_probs"]

        returns = batch["returns"]

        advantages = batch["advantages"]

        action_masks = batch["action_masks"]

        ##########################################################
        # Flatten Actor Inputs
        ##########################################################

        T = obs.shape[0]

        obs = obs.reshape(
            T * NUM_AGENTS,
            OBS_DIM,
        )

        actions = actions.reshape(
            T * NUM_AGENTS,
        )

        old_log_probs = old_log_probs.reshape(
            T * NUM_AGENTS,
        )

        returns = returns.reshape(
            T * NUM_AGENTS,
        )

        advantages = advantages.reshape(
            T * NUM_AGENTS,
        )

        action_masks = action_masks.reshape(
            T * NUM_AGENTS,
            -1,
        )

        ##########################################################
        # Critic Inputs
        ##########################################################

        global_obs = global_obs.repeat_interleave(
            NUM_AGENTS,
            dim=0,
        )
        agent_ids = torch.arange(
            NUM_AGENTS,
            device=self.device,
        ).repeat(T)
        if NORMALIZE_ADVANTAGES:
            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std() + 1e-8
            )
        if self.value_norm is not None:
            self.value_norm.update(returns)

        ##########################################################
        # Statistics
        ##########################################################

        actor_loss_epoch = 0.0
        critic_loss_epoch = 0.0
        entropy_epoch = 0.0

        ##########################################################
        # Number of Samples
        ##########################################################

        dataset_size = obs.shape[0]

        ##########################################################
        # PPO Epochs
        ##########################################################

        for epoch in range(UPDATE_EPOCHS):

            permutation = torch.randperm(
                dataset_size,
                device=self.device,
            )

            ######################################################

            for start in range(
                0,
                dataset_size,
                MINIBATCH_SIZE,
            ):

                end = start + MINIBATCH_SIZE

                idx = permutation[start:end]

                ##################################################
                # Mini-batch
                ##################################################

                mb_obs = obs[idx]

                mb_global = global_obs[idx]

                mb_actions = actions[idx]

                mb_old_log_probs = old_log_probs[idx]

                mb_returns = returns[idx]
                if self.value_norm is not None:
                    normalized_returns = self.value_norm.normalize(
                        mb_returns,
                    )
                else:
                    normalized_returns = mb_returns

                mb_advantages = advantages[idx]

                mb_action_masks  = action_masks[idx]
                mb_agent_ids = agent_ids[idx]

                ##################################################
                # Forward Pass
                ##################################################

                new_log_probs, entropy, values = \
                    self.evaluate_actions(

                        mb_obs,

                        mb_global,

                        mb_actions,
                        mb_agent_ids,

                        mb_action_masks ,
                    )

                ##################################################
                # PPO Ratio
                ##################################################

                ratio = torch.exp(
                    new_log_probs
                    - mb_old_log_probs
                )

                ##################################################
                # Clipped Objective
                ##################################################

                surrogate1 = (
                    ratio
                    * mb_advantages
                )

                surrogate2 = (
                    torch.clamp(
                        ratio,
                        1.0 - PPO_CLIP,
                        1.0 + PPO_CLIP,
                    )
                    * mb_advantages
                )

                ##################################################
                # Losses
                ##################################################

                actor_loss = -torch.min(
                    surrogate1,
                    surrogate2,
                ).mean()

                critic_loss = F.mse_loss(
                    values,
                    normalized_returns,
                )

                entropy_loss = entropy.mean()

                ##################################################
                # Store statistics
                ##################################################

                actor_loss_epoch += actor_loss.item()

                critic_loss_epoch += critic_loss.item()

                entropy_epoch += entropy_loss.item()
                ##################################################
            # Total Loss
            ##################################################

                total_loss = (
                    actor_loss
                    + VALUE_LOSS_COEF * critic_loss
                    - ENTROPY_COEF * entropy_loss
                )

                ##################################################
                # Zero Gradients
                ##################################################

                self.actor_optimizer.zero_grad()

                self.critic_optimizer.zero_grad()

                ##################################################
                # Backpropagation
                ##################################################

                total_loss.backward()

                ##################################################
                # Gradient Clipping
                ##################################################

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )

                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )
                # step
                self.actor_optimizer.step()
                self.critic_optimizer.step()
            
            # Average statistics
            

        # num_updates = (UPDATE_EPOCHS* ((dataset_size + MINIBATCH_SIZE - 1) MINIBATCH_SIZE))
        num_updates = (UPDATE_EPOCHS* ((dataset_size + MINIBATCH_SIZE - 1)// MINIBATCH_SIZE))
        training_stats = {"actor_loss":actor_loss_epoch / num_updates,"critic_loss":critic_loss_epoch / num_updates,"entropy":entropy_epoch / num_updates,}
        return training_stats

    