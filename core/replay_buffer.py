import threading
import numpy as np
from core.her import her_sampler
import random
import core.mod_utils as utils
from scipy.spatial import distance
import torch

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    #def __init__(self, env_params, buffer_size, sample_func):
    def __init__(self, buffer_size):
        #self.env_params = env_params
        #self.T = env_params['max_timesteps']
        self.size = buffer_size // 100
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = her_sampler('', 4)
        # create the buffer to store info
        self.win_states = []
        self.buffers = {'obs': [],
                        'ag': [],
                        'g': [],
                        'actions': [],
                        'r': [],
                        'd': []
                        }
        # thread lock
        #self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        #mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs, mb_ag, mb_actions, reward, done, won = episode_batch
        if won == 1:
            self.win_states.append(mb_ag[-1])
            #print("won", mb_ag)
            mb_g = [mb_ag[-1] for _ in range(len(mb_obs))]
        else:
            goal = random.sample(self.win_states, 1)
            mb_g = [goal for _ in range(len(mb_obs))]
        batch_size = len(mb_obs)
        #with self.lock:
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'].extend(mb_obs)
        self.buffers['ag'].extend(mb_ag)
        self.buffers['g'].extend(mb_g)
        self.buffers['actions'].extend(mb_actions)
        self.buffers['r'].extend(reward)
        #print('d', done)
        self.buffers['d'].extend(done)
        self.n_transitions_stored += len(mb_obs)
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        """
        temp_buffers = {}
        print("win states len: ", len(self.win_states))
        #with self.lock:
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        print('', 'temp_buff')
        print(len(temp_buffers['obs']))
        print(len(temp_buffers['obs'][1:]))
        temp_buffers['obs_next'] = temp_buffers['obs']
        temp_buffers['ag_next'] = temp_buffers['ag']
        # sample transitions
        transitions = self.sample_func.sample_her_transitions(temp_buffers, batch_size)
        """
        #print("sample ")
        #print(self.n_transitions_stored)
        transitions_ind = np.random.random_integers(low=0, high=(self.n_transitions_stored-1), size=batch_size)
        #print(max(transitions_ind))
        #print(len(transitions_ind))
        #print(len(self.buffers['d']))
        alt_goals_ind = np.random.random_integers(low=0, high=self.n_transitions_stored-1, size=batch_size//2)
        """
        transitions = {}
        for key in self.buffers.keys():
            if key == 'g':
                transitions[key] = [self.buffers[key][i] for i in alt_goals_ind]
            else:
                transitions[key] = [self.buffers[key][i] for i in transitions_ind]
        """
        #print("next", self.buffers['ag'])
        #print("goals", self.buffers['g'])
        #print("wins", self.win_states)
        transitions = []
        for i in range(batch_size):
            #print('tr')
            
            if i < batch_size//2:
                state = self.buffers['obs'][transitions_ind[i]]
                action = self.buffers['actions'][transitions_ind[i]]
                next_state = self.buffers['ag'][transitions_ind[i]]
                goal = self.buffers['g'][transitions_ind[i]]
                done = self.buffers['d'][transitions_ind[i]]
                #print((goal))
                #print((next_state))
                #goal = goal.tolist()
                
                if type(goal) == list:
                    goal = np.array(goal[0])
                    #print('hmmm',goal)
                    goal = utils.to_tensor(goal)
                #print((goal))
                #print((next_state))
                """  
                else:
                    pass
                goal = utils.to_tensor(utils.to_numpy(goal.tolist()))
                next_state = utils.to_tensor(utils.to_numpy(next_state.tolist()))
                reward = np.linalg.norm(next_state - goal)
                """
                reward = distance.euclidean(next_state, goal)
                #print(type(reward))
                transitions.append([state, action, next_state, reward, done])
            else:
                state = self.buffers['obs'][transitions_ind[i]]
                action = self.buffers['actions'][transitions_ind[i]]
                next_state = self.buffers['ag'][transitions_ind[i]]
                #print(batch_size//2)
                #print(i)
                #print(max(alt_goals_ind))
                #print(len(self.buffers['g']))
                goal = self.buffers['g'][alt_goals_ind[i-(batch_size//2)]]
                done = self.buffers['d'][transitions_ind[i]]
                #print('goal', goal)
                #print(type(goal))
                if type(goal) == list:
                    goal = utils.to_numpy(goal[0])
                    #print('hmmm',goal)
                    goal = utils.to_tensor(goal)

                reward = distance.euclidean(next_state, goal)
                #reward = np.array([distance])
                #reward = utils.to_tensor(reward)
                #reward = np.linalg.norm(next_state - goal)
                transitions.append([state, action, next_state, torch.tensor(reward), done])


        #transitions = {k: [self.buffers[k][i] for i in transitions_ind] for k in self.buffers.keys()}

        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
