import gym
import my_envs
import numpy as np
from gym import envs
import time

env_key = "v10" #Unique identifier for custom envs (case sensitive)
env_name = 'InvertedPendulum-v2'

env_ids = [spec.id for spec in envs.registry.all()]
test_env_names = [env_name] + [x for x in env_ids if str(env_key) in x] #Returns a list of environment names matching identifier

#print(test_env_names)
#['InvertedPendulum-v2', 
#'InvertedPendulumModified-base-v10', 
#'InvertedPendulumModified-multi-v10', 
#'InvertedPendulumModified-motor-v10', 
#'InvertedPendulumModified-friction-v10', 
#'InvertedPendulumModified-tilt-v10', 
#'InvertedPendulumModified-inertia-v10', 
#'InvertedPendulumModified-mass-v10']


current_env = 'InvertedPendulumModified-multi-v10'
env = gym.make(current_env)
env.reset()

for i in range(100):
	env.step(env.action_space.sample())
	env.render()
	time.sleep(0.03)