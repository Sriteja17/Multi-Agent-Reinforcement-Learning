"""
env.py

Environment interface for MAPPO.

This file is the ONLY place that interacts with CybORG.
The rest of the project only imports CC4Env.
"""

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE

from .config import EPISODE_LENGTH


class CC4Env:

    def __init__(self,red_agent_class=FiniteStateRedAgent):

        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=red_agent_class,
            steps=EPISODE_LENGTH,
        )

        cyborg = CybORG(scenario_generator=scenario)

        # Official CC4 MARL wrapper
        self.env = EnterpriseMAE(cyborg)

        self.agent_names = list(self.env.agents)

    ############################################################

    def reset(self, seed=None):

        observations, info = self.env.reset(seed=seed)

        return observations, info

    ############################################################

    def step(self, actions, messages=None):

        if messages is None:
            messages = {}

        observations, rewards, terminated, truncated, info = \
            self.env.step(actions, messages)

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    

    @property
    def agents(self):

        return self.env.agents

    

    @property
    def possible_agents(self):

        return self.env.possible_agents



    def observation_space(self, agent):

        return self.env.observation_space(agent)

    def action_space(self, agent):

        return self.env.action_space(agent)

    def sample_actions(self):
        """
        Random action for every blue agent.
        Useful for testing.
        """

        actions = {}

        for agent in self.agents:
            actions[agent] = self.action_space(agent).sample()

        return actions

    def get_observation_dims(self):
        dims = {}
        for agent in self.agents:
            dims[agent] = self.observation_space(agent).shape[0]

        return dims

    ############################################################

    def get_action_dims(self):

        dims = {}

        for agent in self.agents:
            dims[agent] = self.action_space(agent).n

        return dims

from ray.tune.registry import register_env


def env_creator(env_config=None):
    """
    RLlib environment creator.
    """
    return CC4Env()


def register_cc4_env():
    """
    Register the environment with RLlib.
    Safe to call multiple times.
    """
    register_env("CC4", lambda config: env_creator(config))