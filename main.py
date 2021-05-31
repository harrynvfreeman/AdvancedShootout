import gym
import gym_advancedshootout
from game.agent.random_agent import RandomAgent
from game.agent.human_agent import HumanAgent
from game.agent.smart_agent import SmartAgent
import copy

env_type = "random"
agent_type = "smart"
safe_guard = 1000

if env_type == "random":
    env = gym.make('AdvancedShootoutRandom-v0')
elif env_type == "smart":
    env = gym.make('AdvancedShootoutSmart-v0')

if agent_type == "random":
    agent = RandomAgent("Player0")
elif agent_type == "human":
    agent = HumanAgent("Player0")
elif agent_type == "smart":
    agent = SmartAgent("Player0")
    

done = False
move_count = 0

observation = env.get_observation();
print('')
while (not done) and (move_count < safe_guard):
    move_count = move_count + 1
    
    action = agent.get_next_action(observation)
    agent.make_action(action)
    observation, reward, done, _ = env.step(action)
    print(observation.name + " did " + str(observation.last_action) + " and has " + str(observation.num_bullets) + " bullets")
    print(agent.name + " did " + str(agent.last_action) + " and has " + str(agent.num_bullets) + " bullets")
    print('')

print('')
print('')
print('Num moves was: ' + str(move_count))

if not done:
    print("Game did not end")
elif reward == 1:
    print(agent.name + " won!")
elif reward == -1:
    print(observation.name + " won!")
else:
    print("Why is reward 0???")