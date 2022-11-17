import random
from collections import namedtuple
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done', 'won'))


class ReplayMemory(object):

    def __init__(self, capacity, sample_type):
        self.capacity = capacity
        self.memory = []
        #self.memory = {'state': [],
        #                'action': [],
        #                'next_state': [],
        #                'reward': [],
        #                'done': []}
        self.position = 0
        self.sample_type = sample_type

        #if self.sample_type == 'HER':
        #    self.her_module = her_sampler(replay_strategy='final', replay_k=6)
        #    self.sample_func = self.her_module.sample_her_transitions

    def push(self, *args):
        """Saves a transition."""
        #print("mem")
        #print(len(self.memory))
        #print('cap', self.capacity)
        if len(self.memory) < self.capacity:
            #print("huh")
            self.memory.append(None)
            #print(None in self.memory)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        if self.sample_type == 'HER':
            batch_size /= 2
            print(self.memory)
            batch = random.sample(self.memory, batch_size)
            print(batch)
            #for trans in 

            return self.sample_func(self.memory, batch_size)
        """
        #print("sample")
        for mem in self.memory:
            if mem == None:
                print("none??")
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        #print(episode_batch)
        #print(type(episode_batch))
        print(episode_batch[0])
        print(episode_batch[0][1])
        #print(type(episode_batch[0]))
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
