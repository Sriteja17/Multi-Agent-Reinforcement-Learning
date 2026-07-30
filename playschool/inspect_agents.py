from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE


def line():
    print("=" * 80)


def section(title):
    print()
    line()
    print(title)
    line()


def main():
    section("Creating CC4 Environment")

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
    )

    cyborg = CybORG(scenario_generator=sg)

    env = EnterpriseMAE(cyborg)

    observations, info = env.reset()

    print("[SUCCESS] Environment Ready")

    section("BLUE AGENTS")

    print(f"Total Blue Agents : {len(env.agents)}")

    for i, agent in enumerate(env.agents):

        print(f"\nAgent {i}")
        print(f"Name : {agent}")

    section("DETAILED AGENT INFORMATION")

    for agent in env.agents:

        line()
        print(agent.upper())
        line()

        print("\nProtected Subnets\n")

        subnets = env.subnets(agent)

        for subnet in subnets:
            print("  •", subnet)

        print("\nProtected Hosts\n")

        hosts = env.hosts(agent)

        for host in hosts:
            print("  •", host)
        print("\nObservation")

        obs = observations[agent]

        print(f"Shape      : {obs.shape}")
        print(f"Dimensions : {len(obs)}")
        print("\nAction Space")

        action_space = env.action_space(agent)

        print(action_space)

        print(f"Total Actions : {action_space.n}")
    section("SUMMARY")

    print()

    for agent in env.agents:

        print(
            f"{agent:15s}"
            f"  Hosts : {len(env.hosts(agent)):2d}"
            f"   Subnets : {len(env.subnets(agent)):2d}"
            f"   Observation : {len(observations[agent]):3d}"
            f"   Actions : {env.action_space(agent).n:3d}"
        )

    print()

    line()
    print("Environment inspection completed successfully.")
    line()


if __name__ == "__main__":
    main()