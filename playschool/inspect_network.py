from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


def line():
    print("=" * 100)


def section(title):
    print()
    line()
    print(title)
    line()


def main():

    ############################################################

    section("CREATING CC4 ENVIRONMENT")

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
    )

    cyborg = CybORG(scenario_generator=sg)

    cyborg.reset()

    state = cyborg.environment_controller.state

    ############################################################

    section("MISSION")

    print(f"Mission Phase : {state.mission_phase}")

    ############################################################

    section("SUBNETS")

    print(f"Total Subnets : {len(state.subnets)}\n")

    for subnet in state.subnets.values():

        print("-" * 80)

        print(f"Subnet Name : {state.subnets_cidr_to_name[subnet.cidr]}")
        print(f"CIDR        : {subnet.cidr}")

        hosts = []

        for hostname, subnet_name in state.hostname_subnet_map.items():

            if subnet_name == state.subnets_cidr_to_name[subnet.cidr]:
                hosts.append(hostname)

        print(f"Hosts ({len(hosts)}):")

        for h in sorted(hosts):
            print("   •", h)

    ############################################################

    section("HOSTS")

    print(f"Total Hosts : {len(state.hosts)}")

    for hostname, host in state.hosts.items():

        print()
        line()

        print(hostname)

        line()

        print("IP Address")

        print(" ", state.hostname_ip_map[hostname])

        print()

        print("Subnet")

        print(" ", state.hostname_subnet_map[hostname])

        print()

        print("Processes")

        if len(host.processes) == 0:
            print("  None")
        else:
            for p in host.processes:
                try:
                    print(f"  PID {p.pid:5d}   {p.process_name}")
                except:
                    print(" ", p)

        print()

        print("Sessions")

        empty = True

        for agent, sessions in host.sessions.items():

            if len(sessions):

                empty = False

                print(f"  {agent}")

                for sid in sessions:
                    print(f"     Session {sid}")

        if empty:
            print("  None")

    ############################################################

    section("HOST -> IP")

    for host, ip in state.hostname_ip_map.items():
        print(f"{host:45s} {ip}")

    ############################################################

    section("HOST -> SUBNET")

    for host, subnet in state.hostname_subnet_map.items():
        print(f"{host:45s} {subnet}")

    ############################################################

    section("BLUE SESSIONS")

    for agent in state.sessions:

        if not agent.startswith("blue"):
            continue

        print()
        print(agent)

        if len(state.sessions[agent]) == 0:
            print("   None")
            continue

        for sid, session in state.sessions[agent].items():

            print(
                f"   Session {sid:2d}"
                f"  Host : {session.hostname}"
            )

    ############################################################

    section("RED SESSIONS")

    for agent in state.sessions:

        if not agent.startswith("red"):
            continue

        print()
        print(agent)

        if len(state.sessions[agent]) == 0:
            print("   None")
            continue

        for sid, session in state.sessions[agent].items():

            print(
                f"   Session {sid:2d}"
                f"  Host : {session.hostname}"
            )

    ############################################################

    section("GREEN SESSIONS")

    count = 0

    for agent in state.sessions:

        if not agent.startswith("green"):
            continue

        count += 1

        print()

        print(agent)

        if len(state.sessions[agent]) == 0:
            print("   None")
            continue

        for sid, session in state.sessions[agent].items():

            print(
                f"   Session {sid:2d}"
                f"  Host : {session.hostname}"
            )

    print()

    print(f"Total Green Agents : {count}")

    ############################################################

    section("NETWORK CONNECTIVITY")

    print(f"Connected Components : {len(state.connected_components)}\n")

    for i, component in enumerate(state.connected_components):

        print(f"Component {i}")

        for host in sorted(component):
            print("   •", host)

        print()

    ############################################################

    section("NETWORK GRAPH")

    print(f"Nodes : {state.link_diagram.number_of_nodes()}")
    print(f"Edges : {state.link_diagram.number_of_edges()}")

    print()

    for edge in sorted(state.link_diagram.edges()):

        print(f"{edge[0]}  <------->  {edge[1]}")

    ############################################################

    section("SUMMARY")

    print(f"Mission Phase : {state.mission_phase}")
    print(f"Hosts         : {len(state.hosts)}")
    print(f"Subnets       : {len(state.subnets)}")
    print(f"Blue Agents   : {len([a for a in state.sessions if a.startswith('blue')])}")
    print(f"Red Agents    : {len([a for a in state.sessions if a.startswith('red')])}")
    print(f"Green Agents  : {len([a for a in state.sessions if a.startswith('green')])}")

    line()
    print("NETWORK INSPECTION COMPLETE")
    line()


if __name__ == "__main__":
    main()