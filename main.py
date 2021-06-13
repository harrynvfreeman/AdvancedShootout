import gym
import gym_advancedshootout
from gym_advancedshootout.envs.advancedshootout_smart_env import AdvancedShootoutSmartEnv
from game.agent.random_agent import RandomAgent
from game.agent.human_agent import HumanAgent
from game.agent.smart_agent import SmartAgent
from game.agent.dumb_agent import DumbAgent
from game.move import Move
import copy
import numpy as np

env_type = "smart"
env_version = None
agent_type = "human"
agent_version = None
deterministic = False
safe_guard = 100
num_iterations = 1000

if env_type == "random":
    env = gym.make('AdvancedShootoutRandom-v0')
elif env_type == "smart":
    AdvancedShootoutSmartEnv.set_deterministic(deterministic)
    AdvancedShootoutSmartEnv.set_version(env_version)
    env = gym.make('AdvancedShootoutSmart-v0')

if agent_type == "random":
    agent = RandomAgent("Player0")
elif agent_type == "human":
    agent = HumanAgent("Player0")
    num_iterations = 1
elif agent_type == "smart":
    if agent_version is None:
        agent_version = np.load('./train/version.npy')
    agent_P_path = './train/' + str(agent_version) + '/P.npy'
    agent_max_bullets_path = './train/' + str(agent_version) + '/max_bullets.npy'
    agent = SmartAgent(agent_P_path, agent_max_bullets_path, deterministic, "Player0")
elif agent_type == "dumb":
    agent = DumbAgent("Player0")
    
uncomplete_count = 0
agent_win_count = 0
env_win_count = 0
incorrect_state_count = 0
move_counts = []
agent_num_bullets = []
opponent_num_bullets = []

agent_move_type_count = {Move.SHIELD: 0,
                         Move.RELOAD: 0,
                         Move.SHOOT: 0,
                         Move.SHOTGUN: 0,
                         Move.ROCKET: 0,
                         Move.SONIC_BOOM: 0}

env_move_type_count = {Move.SHIELD: 0,
                       Move.RELOAD: 0,
                       Move.SHOOT: 0,
                       Move.SHOTGUN: 0,
                       Move.ROCKET: 0,
                       Move.SONIC_BOOM: 0}

for i in range(num_iterations):
    env.reset()
    agent.reset()
    done = False
    move_count = 0

    observation = env.get_observation();
    #print('')
    while (not done) and (move_count < safe_guard):
        move_count = move_count + 1
        
        action = agent.get_next_action(observation)
        agent.make_action(action)
        observation, reward, done, _ = env.step(action)
        
        agent_move_type_count[agent.last_action] = agent_move_type_count[agent.last_action] + 1
        env_move_type_count[observation.last_action] = env_move_type_count[observation.last_action] + 1
        
        if agent_type == "human":
            print(observation.name + " did " + str(observation.last_action) + " and has " + str(observation.num_bullets) + " bullets")
            print(agent.name + " did " + str(agent.last_action) + " and has " + str(agent.num_bullets) + " bullets")
            print('')
            
    move_counts.append(move_count)
    agent_num_bullets.append(agent.num_bullets)
    opponent_num_bullets.append(observation.num_bullets)

    if not done:
        #print("Game did not end")
        uncomplete_count = uncomplete_count + 1
    elif reward == 1:
        agent_win_count = agent_win_count + 1
        #print(agent.name + " won!")
    elif reward == -1:
        env_win_count = env_win_count + 1
        #print(observation.name + " won!")
    else:
        incorrect_state_count = incorrect_state_count + 1
        #print("Why is reward 0???")

print("Incorrect state count is: " + str(incorrect_state_count))
print("Uncompleted count is: " + str(uncomplete_count))
print("Agent win count is: " + str(agent_win_count))
print("Env win count is: " + str(env_win_count))

print('')

print(agent_move_type_count)
print(env_move_type_count)
