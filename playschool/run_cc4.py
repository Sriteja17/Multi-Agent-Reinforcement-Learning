from pprint import pprint
from CybORG import CybORG
print("Great things takes time :/")
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
print("import 3")

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
print("import 4")


def banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def main():

    banner("STEP 1 : Creating Enterprise Scenario")

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
    )

    print("[SUCCESS] Scenario Generator Created")
    banner("STEP 2 : Creating CybORG Simulator")

    cyborg = CybORG(scenario_generator=sg)

    print("[SUCCESS] CybORG Simulator Created")
    banner("STEP 3 : Wrapping with EnterpriseMAE")

    env = EnterpriseMAE(cyborg)

    print("[SUCCESS] EnterpriseMAE Wrapper Loaded")

    banner("STEP 4 : Resetting Environment")

    observations, info = env.reset()

    print("[SUCCESS] Environment Reset")
    banner("STEP 5 : Available Agents")

    print(f"Total Agents : {len(env.agents)}\n")

    for agent in env.agents:
        print(agent)
    banner("STEP 6 : Observation Spaces")

    for agent in env.agents:

        print(f"\n{agent}")

        print("Observation Space :")
        print(env.observation_space(agent))

        print("Action Space :")
        print(env.action_space(agent))
    banner("STEP 7 : Observation Shapes")

    for agent, obs in observations.items():

        print(f"\n{agent}")

        print("Type :", type(obs))

        try:
            print("Shape :", obs.shape)
        except Exception:
            print("Length :", len(obs))
    banner("STEP 8 : Sampling Random Actions")

    actions = {}

    for agent in env.agents:

        action = env.action_space(agent).sample()

        actions[agent] = action

        print(f"{agent:15s} -> {action}")
    banner("STEP 9 : Executing One Step")

    observations, rewards, terminated, truncated, info = env.step(actions)

    print("[SUCCESS] Environment Step Executed")
    banner("STEP 10 : Rewards")

    pprint(rewards)
    banner("STEP 11 : Terminated")

    pprint(terminated)

    banner("STEP 12 : Truncated")

    pprint(truncated)

    banner("STEP 13 : Info")

    pprint(info)
    banner("Nak antha artham aypoindhi (project wont be finished)")
    print("[SUCCESS] CC4 is running correctly!")
    print("[SUCCESS] EnterpriseMAE is working!")
    print("[SUCCESS] Blue agents are interacting with the simulator!")
    print("[SUCCESS] Ready for PPO / MAPPO implementation.")


if __name__ == "__main__":
    main()