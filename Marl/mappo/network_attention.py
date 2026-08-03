"""
networks.py

Neural networks for MAPPO.

Architecture
------------
Shared Actor:
    Observation -> Action Probabilities, via attention across the
    ENTITIES inside a single agent's own observation (mission phase,
    each HQ-subnet block, and the message block).

Central Critic:
    Global Observation -> State Value (per agent), computed via
    attention ACROSS the 5 agents' embeddings.

Notes
-----
The actor used to run MultiheadAttention on a sequence of length 1
(a single unsqueezed token attending to itself). With seq_len == 1,
softmax has nothing to normalize over except the one token, so the
attention weight was always 1.0 and the block degenerated into a
plain linear projection -- it added parameters and compute without
adding any representational power.

BlueFlatWrapper.observation_change (CybORG) shows the observation is
NOT one undifferentiated vector -- it's built by concatenating:

    [mission_phase] + [subnet_0 block] + [subnet_1 block] + [subnet_2 block] + [messages]

where each subnet block is itself

    [subnet one-hot | blocked-subnets mask | comms-policy mask | process-alert bits | connection-alert bits]

and BlueFlatWrapper pads every agent up to blue_agent_4's 3-subnet
"long" observation so the shared actor always sees the same shape.

That gives the actor a real sequence of >1 semantically distinct
entities to attend over (mission, 3 subnet blocks, messages), so
attention here is no longer a no-op -- e.g. the message token can
inform how a subnet block is read, and one subnet's alert state can
inform how another subnet's block is interpreted (lateral movement
between subnets shows up exactly like this).

CAUTION: SUBNET_BLOCK_DIM below assumes each of the NUM_HQ_SUBNETS
blocks reserves a uniform MAX_HOSTS-wide slot for process/connection
alerts, mirroring BlueFlatWrapper._get_init_obs_spaces. If the real
per-subnet host counts used by observation_change aren't uniform
across the 3 HQ subnets, this slicing will be wrong even though the
assert below (which only checks the *total* length) still passes.
Verify with a live env before trusting this in training -- see the
snippet in chat.
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

# Same block layout BlueFlatWrapper uses to build the padded ("long")
# observation -- imported straight from the wrapper so this can't
# silently drift out of sync with it.
from CybORG.Agents.Wrappers.BlueFlatWrapper import NUM_SUBNETS, NUM_HQ_SUBNETS, MAX_HOSTS
from CybORG.Agents.Wrappers.BlueFixedActionWrapper import NUM_MESSAGES, MESSAGE_LENGTH


# ==========================================================
# Entity layout (mirrors BlueFlatWrapper._get_init_obs_spaces)
# ==========================================================

MISSION_DIM = 1
# subnet one-hot + blocked mask + comms-policy mask + process alerts + connection alerts
SUBNET_BLOCK_DIM = 3 * NUM_SUBNETS + 2 * MAX_HOSTS
MESSAGE_DIM = NUM_MESSAGES * MESSAGE_LENGTH
NUM_ENTITY_TOKENS = 1 + NUM_HQ_SUBNETS + 1  # mission + subnet blocks + messages

_EXPECTED_OBS_DIM = MISSION_DIM + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM + MESSAGE_DIM
assert _EXPECTED_OBS_DIM == OBS_DIM, (
    f"Entity split ({_EXPECTED_OBS_DIM}) does not match OBS_DIM ({OBS_DIM}) -- "
    "BlueFlatWrapper's per-subnet block size probably isn't a flat MAX_HOSTS "
    "per subnet at runtime. Print actual per-subnet host counts from "
    "BlueFlatWrapper.observation_change / self.hosts(agent_name) and adjust "
    "SUBNET_BLOCK_DIM (or switch to a per-slot list) accordingly."
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
        Local observation (OBS_DIM dims), laid out the way
        BlueFlatWrapper.observation_change produces it:
            [mission] + [subnet_0] + [subnet_1] + [subnet_2] + [messages]

    Output:
        Action logits (ACTION_DIM dims)

    Splits the flat vector back into its constituent entities, embeds
    each to EMBED_DIM, self-attends across them, then concatenates the
    attended tokens and feeds an MLP head. Concatenation (rather than
    mean-pooling) is used so the head keeps each subnet's identity --
    actions are subnet/host specific, so collapsing "subnet 0 looks
    compromised" and "subnet 2 looks compromised" into one averaged
    vector would throw away exactly the information the policy needs
    to pick the right target.
    """

    def __init__(self):

        super().__init__()

        self.mission_embed = nn.Linear(MISSION_DIM, EMBED_DIM)
        self.subnet_embed = nn.Linear(SUBNET_BLOCK_DIM, EMBED_DIM)
        self.message_embed = nn.Linear(MESSAGE_DIM, EMBED_DIM)

        # Learned per-slot positional embedding. Unlike the critic's
        # agent tokens (interchangeable), "subnet block 1" is a fixed,
        # meaningful slot, so the model should be allowed to treat
        # slots differently rather than assuming permutation symmetry.
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_ENTITY_TOKENS, EMBED_DIM))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )
        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.policy_head = build_mlp(NUM_ENTITY_TOKENS * EMBED_DIM, ACTION_DIM)

    def _split_entities(self, observation):
        """(B, OBS_DIM) -> mission (B,1), subnets (B, NUM_HQ_SUBNETS, SUBNET_BLOCK_DIM), messages (B, MESSAGE_DIM)."""

        subnets_start = MISSION_DIM
        subnets_end = MISSION_DIM + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM

        mission = observation[:, :MISSION_DIM]
        subnets = observation[:, subnets_start:subnets_end].view(
            -1, NUM_HQ_SUBNETS, SUBNET_BLOCK_DIM
        )
        messages = observation[:, subnets_end:]

        return mission, subnets, messages

    def forward(self, observation):
        squeeze_output = observation.dim() == 1
        if squeeze_output:
            observation = observation.unsqueeze(0)

        batch_size = observation.shape[0]

        mission, subnets, messages = self._split_entities(observation)

        mission_tok = torch.relu(self.mission_embed(mission)).unsqueeze(1)    # (B, 1, E)
        subnet_tok = torch.relu(self.subnet_embed(subnets))                    # (B, NUM_HQ_SUBNETS, E)
        message_tok = torch.relu(self.message_embed(messages)).unsqueeze(1)    # (B, 1, E)

        tokens = torch.cat([mission_tok, subnet_tok, message_tok], dim=1)      # (B, NUM_ENTITY_TOKENS, E)
        tokens = tokens + self.pos_embed

        # -------------------------------------------------
        # Multi-Head Self Attention across entities
        # -------------------------------------------------

        attn_out, attention_weights = self.attention(tokens, tokens, tokens)

        # Save attention weights (useful for visualization later, e.g.
        # "did the policy attend to subnet 1 when messages flagged it?")
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
        # Action Logits
        # -------------------------------------------------

        flat = x.reshape(batch_size, -1)
        logits = self.policy_head(flat)

        if squeeze_output:
            logits = logits.squeeze(0)

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

        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)

        batch_size = global_state.shape[0]

        # (batch, NUM_AGENTS * OBS_DIM) -> (batch, NUM_AGENTS, OBS_DIM)
        per_agent_obs = global_state.view(batch_size, NUM_AGENTS, OBS_DIM)

        tokens = torch.relu(
            self.agent_embed(per_agent_obs)
        )

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