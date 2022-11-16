import numpy as np, os, time, sys, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch
from core import replay_memory
from core import ddpg as ddpg
import argparse
import copy
from itertools import chain
#import gym_go #stb - Not sure if this is needed


render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2) (gym_go)', required=True)
env_tag = vars(parser.parse_args())['env']

def mod_state(state):

    #print("mod_state: \n", state)
    black_state = state[0]
    #print(black_state)
    black_list = list(chain.from_iterable(black_state))

    white_state = state[1]
    #print(white_state)
    white_list = list(chain.from_iterable(white_state))

    inv_states = state[3]
    #print(inv_states)
    inv_list = list(chain.from_iterable(inv_states))

    opp_passed = state[4][0][0]

    return np.array(black_list + white_list + inv_list + [opp_passed])

def filter_actions(action, invalid):
    #print("filtering")
    #print(len(action[0]))
    invalid = list(chain.from_iterable(invalid))
    # invalid = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #STB: Testing the filter
    action = action.tolist()[0]
    #print(len(action))
    #print("filtering\n", invalid)
    #action = [-1 if invalid[i] == 1 else action[i] ]
    for i in range(len(action)-1):
        if invalid[i] == 1:
            action[i] = -1
        
    #print(action)
    #print(len(action))
    
    return np.array([action])


class Parameters:
    def __init__(self):

        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        elif env_tag ==  'gym-go': self.num_frames = 10000
        else: self.num_frames = 2000000

        #USE CUDA
        #self.is_cuda = True; self.is_memory_cuda = True
        self.is_cuda = False; self.is_memory_cuda = False #STB

        #Sunchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 1
        else: self.synch_period = 10

        #DDPG params
        self.use_ln = True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 7
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        #Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1


        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_ERL/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = utils_ne.SSNE(self.args)

        #Init population
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))

        #Turn off gradients and put in eval mode
        for actor in self.pop: actor.eval()

        #Init RL Agent
        self.rl_agent = ddpg.DDPG(args)
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.ounoise = ddpg.OUNoise(args.action_dim, env_tag=env_tag)

        #Trackers
        self.num_games = 0; self.num_frames = 0; self.gen_frames = None

    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, is_render, is_action_noise=False, store_transition=True):
        total_reward = 0.0

        #print("evaluate")

        #save_state = self.env.reset()
        #state = self.env.reset()
        full_state = self.env.reset()
        if env_tag == 'gym-go':     #STB: flatten state for go board
            #state = mod_state(save_state)
            #state = mod_state(state)
            state = mod_state(np.array(full_state))
        else:
            state = full_state
            pass

        state = utils.to_tensor(state).unsqueeze(0)
        #print(state)
        if self.args.is_cuda: state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if render and is_render: self.env.render()
    
            action = net.forward(state)
            action.clamp(-1,1)            
            action = utils.to_numpy(action.cpu())
            
            if is_action_noise: action += self.ounoise.noise()
            #print("save", save_state)
            if env_tag == 'gym-go': 
                #action = filter_actions(action, save_state[3]) #STB: Filtering invalid moves will save time in training
                #action = filter_actions(action, state[3]) #STB: Filtering invalid moves will save time in training
                action = filter_actions(action, full_state[3]) #STB: Filtering invalid moves will save time in training
            #print("action", action)
            #print(type(action))
            

            next_state, reward, done, info = self.env.step(action.flatten())  #Simulate one step in environment
            
            
            #print("done?", save_state[5])
            #print("\n")
            save_next_state = copy.deepcopy(next_state)
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            save_next_state = utils.to_tensor(mod_state(save_next_state))
            #print(save_next_state)

            #if store_transition: self.add_experience(save_state, action, next_state, reward, done)
            #if store_transition: self.add_experience(state, action, next_state, reward, done)
            if store_transition: self.add_experience(state, action, save_next_state, reward, done)

            #state = np.array(next_state)[0]
            full_state = np.array(next_state)[0]

            if env_tag == 'gym-go':     #STB: flatten state for go board
                #state = mod_state(state)
                state = mod_state(np.array(full_state))
                
                #print("hmmmm")
                #print(state)
                #print(save_state)
            else:
                state = full_state
                pass
            state = utils.to_tensor(state).unsqueeze(0)
            #print("3", save_state[3])
            #print("5", save_state[5])
            if done == 1:
                #print("done")
                break

        if store_transition: self.num_games += 1


        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        self.gen_frames = 0

        ####################### EVOLUTION #####################
        all_fitness = []
        #Evaluate genomes/individuals
        for net in self.pop:
            fitness = 0.0
            for eval in range(self.args.num_evals): fitness += self.evaluate(net, is_render=False, is_action_noise=False)
            #print("net evaluated")
            all_fitness.append(fitness/self.args.num_evals)

        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        #Validation test
        champ_index = all_fitness.index(max(all_fitness))
        test_score = 0.0
        for eval in range(5): test_score += self.evaluate(self.pop[champ_index], is_render=True, is_action_noise=False, store_transition=False)/5.0

        #NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)


        ####################### DDPG #########################
        #DDPG Experience Collection
        self.evaluate(self.rl_agent.actor, is_render=False, is_action_noise=True) #Train

        #DDPG learning step
        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in range(int(self.gen_frames*self.args.frac_frames_train)):
                transitions = self.replay_buffer.sample(self.args.batch_size)
                batch = replay_memory.Transition(*zip(*transitions))
                self.rl_agent.update_parameters(batch)

            #Synch RL Agent to NE
            if self.num_games % self.args.synch_period == 0:
                self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
                self.evolver.rl_policy = worst_index
                print('Synch from RL --> Nevo')

        return best_train_fitness, test_score, elite_index

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')

    #Create Env
    if env_tag == 'gym-go':
        env = utils.NormalizedActions(gym.make('gym_go:go-v0', size=5, komi=0, reward_method='heuristic'))
    else:
        env = utils.NormalizedActions(gym.make(env_tag))
        print("action space shape[0]: " + str(env.action_space.shape[0]))
    print("action space (action_dim): " + str(env.action_space.n))
    
    #parameters.action_dim = env.action_space.shape[0]
    if env_tag == 'gym-go':
        parameters.action_dim = env.action_space
        parameters.state_dim = 3 * env.observation_space.shape[1] * env.observation_space.shape[2] + 1
    else:
        parameters.action_dim = env.action_space.shape[0]
        parameters.state_dim = env.observation_space.shape[0]
    print("state_dim: " + str(env.observation_space.shape))
    
    parameters.env_tag = env_tag
    #Seed
    #env.seed(parameters.seed);
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    #Create Agent
    agent = Agent(parameters, env)
    print('Running', env_tag, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    next_save = 100; time_start = time.time()
    while agent.num_frames <= parameters.num_frames:
        best_train_fitness, erl_score, elite_index = agent.train()
        print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness if best_train_fitness != None else None, ' Test_Score:','%.2f'%erl_score if erl_score != None else None, ' Avg:','%.2f'%tracker.all_tracker[0][1], 'ENV '+env_tag)
        print('RL Selection Rate: Elite/Selected/Discarded', '%.2f'%(agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']),
                                                             '%.2f' % (agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']),
                                                              '%.2f' % (agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']))
        print()
        tracker.update([erl_score], agent.num_games)
        frame_tracker.update([erl_score], agent.num_frames)
        time_tracker.update([erl_score], time.time()-time_start)

        #Save Policy
        if agent.num_games > next_save:
            next_save += 100
            if elite_index != None: torch.save(agent.pop[elite_index].state_dict(), parameters.save_foldername + 'evo_net')
            print("Progress Saved")








