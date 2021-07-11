import gym
import gym_advancedshootout
from gym_advancedshootout.envs.advancedshootout_smart_env import AdvancedShootoutSmartEnv
from game.agent.random_agent import RandomAgent
from game.agent.human_agent import HumanAgent
from game.agent.smart_agent import SmartAgent
from game.agent.dumb_agent import DumbAgent
from game.agent.cheeky_agent import CheekyAgent
from game.agent.mcts_agent import MctsAgent
from game.move import Move
import numpy as np

env_type = "random"
env_version = 15
env_best = True
agent_type = "mcts"
agent_version = 1
agent_best = True
deterministic = False
safe_guard = 100
num_iterations = 1000

if env_type == "random":
    env = gym.make('AdvancedShootoutRandom-v0')
elif env_type == "smart":
    AdvancedShootoutSmartEnv.set_deterministic(deterministic)
    AdvancedShootoutSmartEnv.set_version(env_version)
    AdvancedShootoutSmartEnv.set_best(env_best)
    env = gym.make('AdvancedShootoutSmart-v0')

if agent_type == "random":
    agent = RandomAgent("Player0")
elif agent_type == "human":
    agent = HumanAgent("Player0")
    num_iterations = 1
elif agent_type == "smart":
    if agent_version is None:
        agent_version = np.load('./train/version.npy')
    if agent_best:
        agent_P_path = './train/best/P.npy'
    else:
        agent_P_path = './train/' + str(agent_version) + '/P.npy'
    agent = SmartAgent(agent_P_path, deterministic, "Player0")
elif agent_type == "dumb":
    agent = DumbAgent("Player0")
elif agent_type == "cheeky":
    agent = CheekyAgent("Player0")
elif agent_type == "mcts":
    if agent_version is None:
        agent_version = np.load('./train/version.npy')
    if agent_best:
        agent_best_version = np.load('./train/best/best_version.npy')
        agent_path = './train/best/' + str(agent_best_version)
    else:
        agent_path = './train/' + str(agent_version)
    agent = MctsAgent(agent_path, "Player0")

uncomplete_count = 0
agent_win_count = 0
env_win_count = 0
incorrect_state_count = 0
move_counts = []
agent_num_bullets = []
opponent_num_bullets = []

agent_move_type_count = {Move.SHIELD.name: 0,
                         Move.RELOAD.name: 0,
                         Move.SHOOT.name: 0,
                         Move.SHOTGUN.name: 0,
                         Move.ROCKET.name: 0,
                         Move.SONIC_BOOM.name: 0}

env_move_type_count = {Move.SHIELD.name: 0,
                       Move.RELOAD.name: 0,
                       Move.SHOOT.name: 0,
                       Move.SHOTGUN.name: 0,
                       Move.ROCKET.name: 0,
                       Move.SONIC_BOOM.name: 0}

for i in range(num_iterations):
    env.reset()
    agent.reset()
    done = False
    move_count = 0
    
    observation = env.get_observation();
    while (not done) and (move_count < safe_guard):
        move_count = move_count + 1
        
        action = agent.get_next_action(observation)
        agent.make_action(action)
        observation, reward, done, _ = env.step(action)
        agent.post_move_update(agent.last_action, env.hidden_agent.last_action)
        
        agent_move_type_count[agent.last_action.name] = agent_move_type_count[agent.last_action.name] + 1
        env_move_type_count[observation.last_action.name] = env_move_type_count[observation.last_action.name] + 1
        
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

print('Move stats: ')
print(agent_move_type_count)
print(env_move_type_count)
