import gym
import gym_go
"""
go_env = gym.make('gym_go:go-v0', size=5, komi=0, reward_method='real')

print(go_env.reset())

print(go_env.observation_space)
print(go_env.action_space)
print()
print()
first_action = (2,4)
second_action = (1,1)
third_action = (0,0)
fourth_action = (3,1)
state, reward, done, info = go_env.step(first_action)
print(state)
print(reward)
print()
print()
state, reward, done, info = go_env.step(second_action)
print(state)
print(reward)
print()
print()
state, reward, done, info = go_env.step(third_action)
print(state)
print(reward)
print()
print()
state, reward, done, info = go_env.step(fourth_action)
print(state)
print(reward)
print()
print()
state, reward, done, info = go_env.step(None)
print(state)
print(reward)
print()
print()
go_env.render('terminal')
"""



import numpy as np

def action(action):
    if len(action > 1):
        return interpret_action(action)
    else:
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
    return action
    
def normalize(num):
    num = (num + 1) / 2
    return num

def interpret_action(action):
    actions = action.tolist()[0]
    size = np.sqrt(len(actions)-1)
    print(size)
    #print("normalize")
    actions = [normalize(x) for x in actions]
    #print(actions)
    #print(min(actions), max(actions))
    #print(len(actions))
    best_ind = np.argmax(actions)
    if best_ind == len(actions)-1:
        out == None
    else:
        row_ind = best_ind//size
        col_ind = best_ind % size
        print(best_ind, row_ind, col_ind)
        out = (row_ind, col_ind)
    return out



inp = np.array([[-0.05154605, -0.21573019, -0.29709896,  0.12097207, -0.1691362,   0.06102301,
   0.39092386,  0.17463624, -0.0030174,  -0.1040705,  -0.567135,    0.20310591,
  -0.01398927,  0.49412674,  0.33842462, -0.68169326,  0.30012828,  0.11138839,
  -0.17997593,  0.21494965, -0.02439139, -0.21696639,  0.10920706,  0.24022868,
   0.20781715, -0.5200747 ]])
print(inp)
print(type(inp))
print(len(inp[0]))

out = action(inp)