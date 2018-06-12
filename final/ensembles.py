import math
import random
import os
from common.multiprocessing_env import SubprocVecEnv
from ppo import *
import datetime
import fnmatch

import gym
import my_envs
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class ensemble():
    def __init__(self, base_env_name, import_tags, num_inputs, num_outputs, is_comp = False, debug = False):
        self.base_env_name = base_env_name
        self.import_tags = import_tags
        self.baseline_policies = {}
        self.baseline_policy_list = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_comp = is_comp

        if not self.is_comp:
            self.compensator_policies = None
            self.compensator_policy_list = None
        else:
            self.compensator_policies = {}
            self.compensator_policy_list = []

        self.debug = debug

    def import_policies(self, baseline_file = None):
        """
        Import policy weights from previous training. If using compensators, must specify
        path to weight file for single baseline policy and function will search for compensator weights.
        If using baselines, function searches for weight files
        """
        baseline_folders = {}
        compensator_folders = {}
        if self.is_comp:
            # get compensator folders and weights
            folder_names = os.listdir('./compensator_weights')
            for mod in self.import_tags:
                compensator_folders[mod] = fnmatch.filter(folder_names, self.base_env_name + mod + '*')

            for key, val in compensator_folders.items():
                folder = os.listdir('./compensator_weights/' + str(val[0]))

                if self.debug:
                    print("weight folder:     ", folder)

                weight_file = fnmatch.filter(folder, '*endweights')
                weight_file = weight_file[0]

                if self.debug:
                    print("Weight file:     ", weight_file)

                self.compensator_policies[key] = PPO(self.num_inputs + self.num_outputs, self.num_outputs)  #inputs = dim(state) + dim(action)
                full_weight_file = './compensator_weights/' + str(val[0]) + '/' + weight_file
                self.compensator_policies[key].load_weights(full_weight_file)

            self.baseline_policies['base'] = PPO(self.num_inputs, self.num_outputs)
            self.baseline_policies['base'].load_weights(baseline_file)

            if self.debug:
                print("Imported Compensator Policies:\n")
                print("Baseline:    ")
                print(self.baseline_policies['base'].model)
                for mod, policy in self.compensator_policies.items():
                    print("Modification:    ", mod)
                    print(policy.model)

            self.compensator_policy_list = self.compensator_policies.values()
            self.baseline_policy_list = self.baseline_policies.values()

        else:
            folder_names = os.listdir('./baseline_weights')
            for mod in self.import_tags:
                baseline_folders[mod] = fnmatch.filter(folder_names, self.base_env_name + mod + '*')

            for key, val in baseline_folders.items():
                folder = os.listdir('./baseline_weights/' + str(val[0]))

                if self.debug:
                    print("weight folder:     ", folder)

                weight_file = fnmatch.filter(folder, '*endweights')
                weight_file = weight_file[0]

                if self.debug:
                    print("Weight file:     ", weight_file)

                self.baseline_policies[key] = PPO(self.num_inputs, self.num_outputs)
                full_weight_file = './baseline_weights/' + str(val[0]) + '/' + weight_file
                self.baseline_policies[key].load_weights(full_weight_file)

            if self.debug:
                print("Imported Policies:\n")
                for mod, policy in self.baseline_policies.items():
                    print("Modification:    ", mod)
                    print(policy.model)

            self.baseline_policy_list = self.baseline_policies.values()

    # def uniform_action(self, state):
    #     """ Outputs a uniform average of actions from variable number of policies"""
    #     num_policies = len(self.policy_list)
    #     actions = np.array([policy.model.sample_action(state) for policy in self.policy_list])

    #     action = actions.sum()/num_policies
    #     return action

    def weighted_action(self, state, policy_weights):
        """ Outputs a weighted sum of actions defined by weight vector and variable number of policies"""
        if self.is_comp:
            # TODO: need to decide if baseline is multiplied by weights or left alone, after that same thing
            if policy_weights.shape[1] != len(self.compensator_policy_list):
                print("Weight vector length must match number of policies\n")
                print("weights:    ", policy_weights)
                print("length ", policy_weights.shape[1])
                print("policy list:    ", self.compensator_policy_list)
                print("length: ", len(self.compensator_policy_list))
                return

            base_action = self.baseline_policies['base'].model.sample_action(state).squeeze(0)
            state = np.hstack((state, base_action.cpu().numpy()))
            for i, policy in enumerate(self.compensator_policy_list):
                if i == 0:
                    actions = policy.model.sample_action(state)
                else:
                    actions = torch.cat((actions, policy.model.sample_action(state)), 2)

            actions = actions.squeeze(0)
            action_dim = actions.size()[1]/policy_weights.size()[1]
            policy_weights_tiled = policy_weights.view(-1, 1).repeat(1, action_dim).view(policy_weights.size()[0], policy_weights.size()[1]*action_dim)
            if self.debug:
                print("baseline action:")
                print(base_action, base_action.size())
                print("action tensor:")
                print(actions, actions.size())
                
#             comp_action = torch.sum((actions * policy_weights), dim=1).unsqueeze(1)
            comp_action = torch.sum((actions * policy_weights_tiled), dim=1).unsqueeze(1)
            action = base_action + comp_action

            if self.debug:
                print("weights:    ", policy_weights)
                print("length ", policy_weights.shape[1])
                print("comp policy list:    ", self.compensator_policy_list)
                print("length: ", len(self.compensator_policy_list))

                print("multi-policy multi-env compensation action tensor: ")
                print(actions, actions.size())

                print("ensemble weights:")
                print(policy_weights, policy_weights.size())

                print("baseline action:")
                print(base_action, base_action.size())
                print("compensator action:")
                print(comp_action, comp_action.size())
                print("full action:")
                print(action, action.size())

            return action

        else:
            if policy_weights.shape[1] != len(self.baseline_policy_list):
                print("Weight vector length must match number of policies\n")
                print("weights:    ", policy_weights)
                print("length ", policy_weights.shape[1])
                print("policy list:    ", self.baseline_policy_list)
                print("length: ", len(self.baseline_policy_list))
                return

            if self.debug:
                print("weights:    ", policy_weights)
                print("length ", policy_weights.shape[1])
                print("policy list:    ", self.baseline_policy_list)
                print("length: ", len(self.baseline_policy_list))

            for i, policy in enumerate(self.baseline_policy_list):
                if i == 0:
                    actions = policy.model.sample_action(state)
                else:
                    actions = torch.cat((actions, policy.model.sample_action(state)), 2)

            actions = actions.squeeze(0)

            if self.debug:
                print("multi-policy multi-env action tensor: ")
                print(actions, actions.size())
                print("ensemble weights:")
                print(policy_weights, policy_weights.size())

            action = torch.sum((actions * policy_weights), dim=1).unsqueeze(1)
            if self.debug:
                print("action:")
                print(action, action.size())
            return action


class PPO_Ensemble(PPO):
    def __init__(self, num_inputs, num_outputs, ensemble, hidden_size=64, lr=3e-4, num_steps=2048,
                 mini_batch_size=64, ppo_epochs=10, threshold_reward=950, action_appended=False):
        super().__init__(num_inputs, num_outputs, hidden_size=hidden_size, lr=lr, num_steps=num_steps,
                         mini_batch_size=mini_batch_size, ppo_epochs=ppo_epochs, threshold_reward=threshold_reward)
        self.ensemble = ensemble
        self.action_appended = action_appended

    def collect_data(self, envs):
        if self.state is None:
            state = envs.reset()

        #----------------------------------
        #collect data
        #----------------------------------
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0
        counter = 0

        for _ in range(self.num_steps):
            state = torch.FloatTensor(state).to(device)

            if self.action_appended:
                base_action = self.ensemble.baseline_policies['base'].model.sample_action(state.cpu().numpy()).squeeze(0)
                state = torch.cat((state, base_action), dim=1)
                print("action concatenated state:", state)

            dist, value = self.model(state)

            weights = dist.sample()
            action = self.ensemble.weighted_action(state.cpu().numpy(), weights)
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
#             next_state, reward, done, _ = envs.step(action)

            log_prob = dist.log_prob(weights)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(weights)

            state = next_state
            self.frame_idx += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = self.model(next_state)

        return log_probs, values, states, actions, rewards, masks, next_value

class ensemble_testing_envs(testing_envs):
    def __init__(self, env_names, VISUALIZE, COMPENSATION, results_dir, train_env_index, logging_interval=10):
        super().__init__(env_names, VISUALIZE, COMPENSATION,
                         results_dir, train_env_index, logging_interval=10)

    def test_env(self, env, ensemble_net, weight_net, action_append=False, comp_model=None):

        def test_action(state):
            dist, value = weight_net.model(state)
            weights = dist.sample()
            action = ensemble_net.weighted_action(state.cpu().numpy(), weights)
            return action

        state = env.reset()
        if self.vis:
            env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
#             if action_append:
#                 base_action = self.ensemble_net.baseline_policies['base'].model.sample_action(state.cpu().numpy()).squeeze(0)
#                 state = torch.cat((state, base_action), dim=1)

            sample = test_action(state)

            #state = torch.FloatTensor(state).unsqueeze(0).to(device)
            #dist, _ = control_model(state)
            #sample = dist.sample().cpu().numpy()[0]

            next_state, reward, done, _ = env.step(sample.cpu().numpy())
            state = next_state
            if self.vis:
                env.render()
            total_reward += reward
        return total_reward
