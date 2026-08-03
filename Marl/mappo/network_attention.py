"""
networks.py

Neural networks for MAPPO.

Architecture
------------
Shared Actor:
    Observation -> Action Probabilities

Central Critic:
    Global Observation -> State Value (per agent), computed via
    attention ACROSS the 5 agents' embeddings.

Notes
-----
The previous actor ran MultiheadAttention on a sequence of length 1
(a single unsqueezed token attending to itself). With seq_len == 1,
softmax has nothing to normalize over except the one token, so the
attention weight is always 1.0 and the block degenerates into a
plain linear projection -- it added parameters and compute without
adding any representational power.

The critic, on the other hand, is a natural fit for attention: it
already sees all 5 agents' observations at once, so attending across
agents (instead of flatten-and-concat) lets it learn cross-agent
dependencies explicitly (e.g. agent 3's value depends on what agent 1
is observing on a shared subnet).

If OBS_DIM decomposes into per-host / per-subnet blocks, the actor
can be upgraded the same way (attend across entities within a single
agent's observation). Until that structure is confirmed, the actor
is a plain MLP -- same capacity, no wasted no-op attention.
"""

import torch
import torch.nn as nn

from .config import (
    OBS_DIM,
    ACTION_DIM,
    EMBED_DIM,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_AGENTS,
)


# ==========================================================
# Utility
# ==========================================================

def build_mlp(input_dim, output_dim):
    """
    Build a simple feed-forward MLP.
    """

    layers = []

    current = input_dim

    for _ in range(NUM_HIDDEN_LAYERS):

        layers.append(nn.Linear(current, HIDDEN_DIM))
        layers.append(nn.ReLU())

        current = HIDDEN_DIM

    layers.append(nn.Linear(current, output_dim))

    return nn.Sequential(*layers)


# ==========================================================
# Shared Actor
# ==========================================================

class SharedActor(nn.Module):
    """
    One policy shared by ALL blue agents.

    Input:
        Local observation (210 dims)

    Output:
        Action logits (242 dims)

    NOTE: no attention here. A single observation vector is not a
    sequence, so there is nothing meaningful to attend over unless
    OBS_DIM is known to decompose into per-entity (per-host /
    per-subnet) blocks. If it does, reshape the observation into
    (num_entities, features_per_entity) and use the same pattern as
    CentralCritic below.
    """

    def __init__(self):

        super().__init__()

        self.policy = build_mlp(
            OBS_DIM,
            ACTION_DIM,
        )

    def forward(self, observation):
        logits = self.policy(observation)
        return logits


# ==========================================================
# Central Critic
# ==========================================================

class CentralCritic(nn.Module):
    """
    Centralized critic.

    Receives the GLOBAL STATE:

        concat(
            obs0,
            obs1,
            obs2,
            obs3,
            obs4
        )

    Instead of flattening this into one big vector and feeding it to
    an MLP, we reshape it back into (NUM_AGENTS, OBS_DIM) and run
    self-attention ACROSS agents. Each agent's embedding attends to
    every other agent's embedding, so the critic can explicitly
    learn cross-agent value dependencies rather than inferring them
    implicitly through dense weights over a flattened vector.

    Output: one value estimate per agent, shape (batch, NUM_AGENTS).
    """

    def __init__(self):

        super().__init__()

        self.agent_embed = nn.Linear(
            OBS_DIM,
            EMBED_DIM,
        )

        # self.attention = nn.MultiheadAttention(
        #     embed_dim=EMBED_DIM,
        #     num_heads=NUM_HEADS,
        #     batch_first=True,
        # )

        # self.norm = nn.LayerNorm(EMBED_DIM)

        # self.value_head = build_mlp(
        #     EMBED_DIM,
        #     1,
        # )

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        # First LayerNorm
        self.norm1 = nn.LayerNorm(EMBED_DIM)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )

        # Second LayerNorm
        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.value_head = build_mlp(
            EMBED_DIM,
            1,
        )

    def forward(self, global_state):
        # print(global_state.shape)

        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)

        batch_size = global_state.shape[0]

        # (batch, NUM_AGENTS * OBS_DIM) -> (batch, NUM_AGENTS, OBS_DIM)
        per_agent_obs = global_state.view(batch_size, NUM_AGENTS, OBS_DIM)

        # tokens = self.agent_embed(per_agent_obs)  # (B, N, E)
        tokens = torch.relu(
            self.agent_embed(per_agent_obs)
        )

        # attn_out, attention_weights = self.attention(
        #     tokens,
        #     tokens,
        #     tokens,
        # )

        # # residual + norm, standard transformer-block practice
        # x = self.norm(tokens + attn_out)



        # self.norm2 = nn.LayerNorm(EMBED_DIM)

        # values = self.value_head(x).squeeze(-1)  # (B, N)

        # return values
        # -------------------------------------------------
        # Multi-Head Self Attention
        # -------------------------------------------------

        attn_out, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
        )

        # Save attention weights (useful for visualization later)
        self.last_attention = attention_weights.detach()

        # -------------------------------------------------
        # First Residual Block
        # -------------------------------------------------

        x = self.norm1(tokens + attn_out)

        # -------------------------------------------------
        # Feed Forward Network
        # -------------------------------------------------

        ff = self.ffn(x)

        # -------------------------------------------------
        # Second Residual Block
        # -------------------------------------------------

        x = self.norm2(x + ff)

        # -------------------------------------------------
        # Value Prediction
        # -------------------------------------------------

        values = self.value_head(x).squeeze(-1)
        if values.shape[0] == 1:
            values = values.squeeze(0)
        return values



# ==========================================================
# MAPPO Network
# ==========================================================

class MAPPOModel(nn.Module):
    """
    Holds both actor and critic.

    This makes saving/loading checkpoints easier.
    """

    def __init__(self):

        super().__init__()

        self.actor = SharedActor()

        self.critic = CentralCritic()


    def act(self, observation):

        """
        Returns action logits.

        PPO will convert these into a categorical
        distribution.
        """

        return self.actor(observation)


    def evaluate(self, global_state):

        """
        Returns critic value estimate per agent, shape (batch, NUM_AGENTS).
        """

        return self.critic(global_state)