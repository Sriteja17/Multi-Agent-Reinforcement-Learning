from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE

import numpy as np


def line():
    print("=" * 100)


def section(title):
    print()
    line()
    print(title)
    line()


def print_vector(obs):

    print(f"Length : {len(obs)}")
    print(f"Shape  : {obs.shape}")

    print("\nIndex    Value")
    print("-" * 25)

    for i, value in enumerate(obs):
        print(f"{i:4d} --> {value}")


def compare_vectors(before, after):

    changed = []

    for i in range(len(before)):
        if before[i] != after[i]:
            changed.append((i, before[i], after[i]))

    print(f"\nChanged Features : {len(changed)}")

    print("\nIndex    Before    After")
    print("-" * 35)

    for idx, b, a in changed:
        print(f"{idx:4d}      {b:3d}   --->   {a:3d}")


def main():

    section("Creating Environment")

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
    )

    env = EnterpriseMAE(CybORG(scenario_generator=sg))

    observations, info = env.reset()

    ############################################################

    section("INITIAL OBSERVATIONS")

    for agent in env.agents:

        print(f"\n\n{agent.upper()}")

        obs = observations[agent]

        print_vector(obs)

        print("\nStatistics")

        print(f"Minimum : {obs.min()}")
        print(f"Maximum : {obs.max()}")
        print(f"Mean    : {obs.mean():.3f}")

        unique = np.unique(obs)

        print(f"Unique Values : {unique}")

    ############################################################

    section("TAKING ONE RANDOM STEP")

    actions = {}

    for agent in env.agents:
        actions[agent] = env.action_space(agent).sample()

    new_obs, rewards, terminated, truncated, info = env.step(actions)

    ############################################################

    section("OBSERVATION DIFFERENCES")

    for agent in env.agents:

        print(f"\n\n{agent.upper()}")

        compare_vectors(
            observations[agent],
            new_obs[agent]
        )

    ############################################################

    section("DONE")

    print("Observation inspection complete.")


if __name__ == "__main__":
    main()