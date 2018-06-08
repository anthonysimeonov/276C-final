import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class PPO:
    def __init__(self, num_inputs, num_outputs, hidden_size=64, lr = 3e-4, num_steps = 2048,
                 mini_batch_size = 64, ppo_epochs = 10, threshold_reward = 950):
        
        
        #Hyper params:
        self.hidden_size      = hidden_size
        self.lr               = lr
        self.num_steps        = num_steps
        self.mini_batch_size  = mini_batch_size
        self.ppo_epochs       = ppo_epochs
        self.threshold_reward = threshold_reward
        
        #Model params:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.early_stop = False
        self.state = None
        self.frame_idx = 0
        
        self.model = ActorCritic(self.num_inputs, self.num_outputs, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def load_weights(self, directory):
        if use_cuda:
            weights = torch.load(directory)
        else:
            weights = torch.load(directory, map_location='cpu')
        self.model.load_state_dict(weights)

    def save_weights(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, 
                   returns, values, clip_param=0.2):

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantages = returns - values

        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, 
                                                                                  states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    
    def collect_data(self, envs):
        if self.state is None:
            state = envs.reset()
            
        #----------------------------------
        #collect data
        #----------------------------------
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0
        counter = 0

        for _ in range(self.num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = self.model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)

            state = next_state
            self.frame_idx += 1


        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = self.model(next_state)

        return log_probs, values, states, actions, rewards, masks, next_value

    
                
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(self.init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)
            
    def sample_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, value = self.forward(state)
            action = dist.sample()
            return action
            
#Class that takes care of testing/visualizing rollouts and logging data

class testing_envs():
    def __init__(self, env_names, VISUALIZE, COMPENSATION, results_dir, logging_interval = 10):
        self.env_names = env_names
        self.num_test_envs = len(env_names)
        self.results_dir = results_dir
        self.logging_interval = logging_interval
        self.vis = VISUALIZE
        self.comp = COMPENSATION
        self.cmap = plt.cm.get_cmap('Spectral') #easy to see color range
        
        #Plotting figures
        self.subplots = None
        self.overlay = None
        self.overlay_std = None
        
        self.envs = [self.env_init(name) for name in env_names]
        
    def env_init(self, name):
        logdir = ("./" + name + "_videos/")
        env = gym.make(name)
        if self.vis:
            if not os.path.exists(logdir):
                os.mkdir(logdir)
                env = gym.wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: episode_id%self.logging_interval==0)
        return env
    
    def test_env(self, env, control_model, comp_model = None):
        state = env.reset()
        if self.vis: 
            env.render()
        done = False
        total_reward = 0
        while not done:
            
            sample = control_model.sample_action(state)
            
            #state = torch.FloatTensor(state).unsqueeze(0).to(device)
            #dist, _ = control_model(state)
            #sample = dist.sample().cpu().numpy()[0]
                            
            next_state, reward, done, _ = env.step(sample)
            state = next_state
            if self.vis:
                env.render()
            total_reward += reward
        return total_reward
        
    def plot(self, frame_idx, rewards, stds, which_plts, save = 0, save_indx = 'end'):  
        clear_output(True)
        num_plots = self.num_test_envs
        x = range(len(rewards))

        rewards = np.array(rewards) #v-stacks the rewards, each env of rewards is thus a column
        stds = np.array(stds) #v-stacks the standard deviations, each env of standard deviation is thus a column
        rew_high = rewards + stds
        rew_low = rewards - stds
        
        if which_plts[0]:
            self.subplots = plt.figure(figsize=(20,8*num_plots))
            for i in range(num_plots):
                subplot_ax = self.subplots.add_subplot(num_plots, 1, i+1)
                subplot_ax.set_title('env: %s frame %s. reward: %s' % (self.env_names[i], frame_idx, rewards[-1,i]))        
                subplot_ax.plot(x, rewards[:,i],
                         x, rew_high[:,i],
                         x, rew_low[:,i], color=self.cmap((i)/(num_plots)),linewidth=4)
                subplot_ax.fill_between(x, rew_high[:,i], rew_low[:,i], color=self.cmap((i)/(num_plots)),alpha=0.5)

        if which_plts[1]:
            self.overlay = plt.figure(figsize=(20,8))
            overlay_ax = self.overlay.add_subplot(1,1,1)
            overlay_ax.set_title("All rewards, frame %s" % (frame_idx))
            custom_lines = []
            for i in range(num_plots):
                overlay_ax.plot(x,rewards[:,i], color=self.cmap((i)/(num_plots)), label = self.env_names[i],linewidth=4)
                custom_lines.append(Line2D([0], [0], color=self.cmap((i)/(num_plots)), lw=4)) #For Legends
            overlay_ax.legend(custom_lines,self.env_names,loc=2,prop={'size': 11})
             
        if which_plts[2]:
            self.overlay_std = plt.figure(figsize=(20,8))
            overlay_std_ax = self.overlay_std.add_subplot(1,1,1)
            overlay_std_ax.set_title("All rewards, frame %s" % (frame_idx))
            custom_lines = []
            for i in range(num_plots):
                overlay_std_ax.plot(x, rewards[:,i],
                x, rew_high[:,i],
                x, rew_low[:,i], color=self.cmap((i)/(num_plots)),linewidth=4)
                overlay_std_ax.fill_between(x, rew_high[:,i], rew_low[:,i], color=self.cmap((i)/(num_plots)), alpha=0.5)
                custom_lines.append(Line2D([0], [0], color=self.cmap((i)/(num_plots)), lw=4)) #For Legends
            overlay_std_ax.legend(custom_lines,self.env_names,loc=2,prop={'size': 11})

        if save:
            print("Saving figures")
            if which_plts[0]:
                self.subplots.savefig(self.results_dir + 'Reward_subplots_' + save_indx+ '.png')
            if which_plts[1]:
                self.overlay.savefig(self.results_dir + 'Total_rewards_' + save_indx+ '.png')
            if which_plts[2]:
                self.overlay_std.savefig(self.results_dir + 'Total_rewards_std_' + save_indx+ '.png')
                
            np.savez(self.results_dir + 'data_' + save_indx, rewards, stds, np.array(self.env_names), np.array(frame_idx))

        plt.show()
        
        
        
        