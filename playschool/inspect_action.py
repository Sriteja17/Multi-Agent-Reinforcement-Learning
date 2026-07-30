from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE


def line():
    print("=" * 110)


def section(title):
    print()
    line()
    print(title)
    line()


def main():

    ###############################################################

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

    ###############################################################

    for agent in env.agents:

        section(f"{agent.upper()} ACTION SPACE")

        labels = env.action_labels(agent)
        actions = env.actions(agent)
        mask = env.action_mask(agent)

        total_actions = len(labels)
        valid_actions = sum(mask)

        print(f"Total Actions : {total_actions}")
        print(f"Valid Actions : {valid_actions}")
        print(f"Invalid Actions : {total_actions - valid_actions}")

        line()

        print(f"{'ID':<5} {'VALID':<8} {'ACTION LABEL'}")

        line()

        for idx, (label, valid) in enumerate(zip(labels, mask)):

            symbol = "✔" if valid else "✘"

            print(f"{idx:<5} {symbol:<8} {label}")

        ###########################################################

        section(f"{agent.upper()} ACTION OBJECTS")

        print(
            "These are the actual CybORG Action objects that the wrapper "
            "creates from each action ID.\n"
        )

        for idx, action in enumerate(actions):

            print(f"{idx:3d} -> {action}")


    section("SUMMARY")

    for agent in env.agents:

        total = len(env.action_labels(agent))
        valid = sum(env.action_mask(agent))

        print(
            f"{agent:15s}"
            f"  Total: {total:3d}"
            f"   Valid: {valid:3d}"
            f"   Invalid: {total-valid:3d}"
        )

    line()
    print("Action inspection completed successfully.")
    line()


if __name__ == "__main__":
    main()