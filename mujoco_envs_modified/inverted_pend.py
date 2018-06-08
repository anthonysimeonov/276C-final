
import gym
import mujoco_py

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import mujoco_py as mp
import math
import os
import gym
import time
import numpy as np


env = gym.make('InvertedPendulum-v2')
env.reset()

MODEL_XML = """
<mujoco model="inverted_pendulum_base">
	<compiler inertiafromgeom="true"/>
	<default>
		<joint armature="0" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
		<tendon/>
		<motor ctrlrange="-3 3"/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
	<size nstack="3000"/>
	<worldbody>
		<!--geom name="ground" type="plane" pos="0 0 0" /-->
		<geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
		<body name="cart" pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
			<joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" frictionloss="0" type="slide"/>
			<!--geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/-->
			<geom name="cart" pos="0 0 0" quat="0.6 0.7 0.9 0" size="0.1 0.1" type="capsule"/>
			<body name="pole" pos="0 0 0">
				<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
				<joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" frictionloss="0" type="hinge"/>
				<geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
				<body name="ball" pos="0 0 0.61">
					<inertial pos="0 0 0" mass="1" diaginertia="166.667 166.667 166.667"/>
					<geom contype="0" name="ball" pos="0 0 0" size="0.1" type="sphere"/>
				</body>
				<!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
			</body>
		</body>
	</worldbody>
	<actuator>
		<motor gear="100" joint="slider" name="slide"/>
	</actuator>
</mujoco>
"""


model = load_model_from_xml(MODEL_XML)

sim = MjSim(model)

viewer = MjViewer(sim)



while True:
	action = (env.action_space.sample())

	sim.data.ctrl[0] = action[0]
	#sim.data.ctrl[1] = action[0]

	sim.step()
	viewer.render()
