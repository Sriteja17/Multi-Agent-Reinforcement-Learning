"""
config.py

Configuration file for MAPPO training on CAGE Challenge 4 (CC4).
Modify hyperparameters here instead of changing them throughout the code.
"""

# ==========================================================
# Environment
# ==========================================================

NUM_AGENTS = 5

EPISODE_LENGTH = 100          # Must match EnterpriseScenarioGenerator(steps=100)

MISSION_PHASES = 3


# ==========================================================
# Observation / Action Dimensions
# ==========================================================

# Blue Agents 0-3
SMALL_OBS_DIM = 92
SMALL_ACTION_DIM = 82

# Blue Agent 4
LARGE_OBS_DIM = 210
LARGE_ACTION_DIM = 242

# Shared-policy dimensions
#
# We pad every observation to 210 features
# and every action distribution to 242 actions.
#
# Invalid actions are masked before sampling.
#
OBS_DIM = LARGE_OBS_DIM
ACTION_DIM = LARGE_ACTION_DIM


# ==========================================================
# MAPPO Hyperparameters
# ==========================================================

TOTAL_EPISODES = 8000

ROLLOUT_STEPS = 512

UPDATE_EPOCHS = 10

MINIBATCH_SIZE = 256

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

PPO_CLIP = 0.2

VALUE_LOSS_COEF = 0.5

ENTROPY_COEF = 0.01

MAX_GRAD_NORM = 0.5


# ==========================================================
# Neural Network
# ==========================================================

HIDDEN_DIM = 256

NUM_HIDDEN_LAYERS = 2

ACTIVATION = "relu"


# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda"
# ==========================================================
# Logging
# ==========================================================

PRINT_EVERY = 10

SAVE_EVERY = 500

CHECKPOINT_DIR = "checkpoints"

LOG_DIR = "logs"


# ==========================================================
# Randomness
# ==========================================================

SEED = 42
# PPO Training

UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256

VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

PPO_CLIP = 0.2

MAX_GRAD_NORM = 0.5

# Add these
VALUE_CLIP = False          # Optional value clipping
NORMALIZE_ADVANTAGES = True # Standard MAPPO practice