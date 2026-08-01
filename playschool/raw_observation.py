from pprint import pprint

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


def line():
    print("=" * 100)


def main():

    line()
    print("CREATING RAW CYBORG ENVIRONMENT")
    line()

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
    )

    env = CybORG(scenario_generator=sg)

    print("[SUCCESS] Environment Created")

    line()
    print("RESETTING")
    line()

    result = env.reset()

    print(type(result))

    print("\nAvailable attributes:\n")

    print(dir(result))

    line()

    print("Observation object type:\n")

    print(type(result.observation))

    line()

    print("Observation:\n")

    pprint(result.observation)

    line()

    print("Action Space:\n")

    pprint(result.action_space)

    line()

    print("Reward")

    print(result.reward)

    line()

    print("Done")

    print(result.done)

    line()


if __name__ == "__main__":
    main()