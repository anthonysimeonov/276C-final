import gym
import my_envs
import time
import math
import numpy as np
import os
import mujoco_py



start_time = time.time()
#my_envs
env_name = 'ReacherSpringy-v2'
#mujoco_envs
#env_name = 'Reacher-v2'
#env_name = 'HalfCheetah-v2'

#env_name = 'CartPole-v0'


env = gym.make(env_name)
#env.max_episode_steps = 100

#Logging parameters
VISUALIZE = True
logging_interval = 1
logdir = './env_tests_videos/'

if VISUALIZE:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    env = gym.wrappers.Monitor(env, logdir, force=True)
    #env = gym.wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: episode_id%logging_interval==0)


env.reset()
counter = 0

# x = np.hstack((np.ones(500), np.ones(500)*-1))
# for i, val in enumerate(x):
# 	#x = math.sin(time.time()*10)
# 	#x = env.action_space.sample()
# 	env.render()
# 	print(val)

# 	env.step([val, 0])
# 		#counter = counter + 1
# 		#if counter == 10:
# 		#	counter = 0
# 		#	x = x*-1
# 	time.sleep(0.01)



# for i in range(500):
# 	env.render()
# 	action = ([1,0])
# 	observation, reward, done, _ = env.step(action)

# 	if i%10 == 0:
# 		print("Reward is {}, done is {}".format(reward,done))

# 	if done:
# 		print("Final state is iter {}, reward is {}, done is {}".format(i,reward,done))
# 		break


while(1):
	env.render()
	action = env.action_space.sample()

	action = np.ones(2)*np.sin(time.time()*2*np.pi/5)*10
	action[1] = 0
	#print(action)
	observation,reward,done,_ = env.step(action)

	'''if counter%10 == 0:
		print("Reward is {}, done is {}".format(reward,done))

	if done:
		print("Final state is iter {}, reward is {}, done is {}".format(counter,reward,done))
		break
	'''
	print(done)
	if done:
		print('Yay')
		break

	counter = counter + 1
	time.sleep(0.01)



	'''while(1):
		env.render()
		action = 5*(env.action_space.sample())
		print(action)
		observation,reward,done,_ = env.step(action)'''
