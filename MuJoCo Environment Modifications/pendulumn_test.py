import mujoco_py as mp
import math
import os
import gym
import time
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
#-----------------------------------------------------
#This test file runs position control on a kinova arm
#-----------------------------------------------------



#set arm sytstem path
xml = "inverted_pendulum.xml"
path = os.getcwd() + "/" + xml

#load arm
model = mp.load_model_from_path(path)
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)

start_time = sim.data.time
print(start_time)

for i in range(300):

    feedback = sim.data.actuator_velocity[0]
     
    force = 0.1
    sim.data.ctrl[0] = force
    #sim.data.ctrl[-5] = 0
    #sim.data.ctrl[-4] = 0
    #sim.data.ctrl[-3] = 0
    #sim.data.ctrl[-2] = 0
    #sim.data.ctrl[-1] = 0
    sim.step()
    viewer.render()
    #print("cfrc_ext is {}" .format(sim.data.cfrc_int))
