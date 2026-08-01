# ignore this file gng , just random stuff , used to check the env creation of cc4 .
# pranay is tall btw 
import time

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE

print("=" * 70)
print("Creating Environment")
print("=" * 70)

sg = EnterpriseScenarioGenerator(
    blue_agent_class=SleepAgent,
    green_agent_class=EnterpriseGreenAgent,
    red_agent_class=FiniteStateRedAgent,
    steps=100,
)

cyborg = CybORG(scenario_generator=sg)
env = EnterpriseMAE(cyborg)

print("Environment Created")

print("\nResetting...")

t0 = time.time()
obs, info = env.reset()
print(f"Reset Time : {time.time() - t0:.3f} sec")

actions = {
    agent: env.action_space(agent).sample()
    for agent in env.agents
}

print("\nRunning 100 environment steps...")

start = time.time()

for i in range(100):
    obs, reward, terminated, truncated, info = env.step(actions)

elapsed = time.time() - start

print("\n" + "=" * 70)
print(f"100 Steps Time : {elapsed:.3f} sec")
print(f"Average Step   : {elapsed/100:.4f} sec")
print("=" * 70)