from gym.envs.registration import registry, register, make, spec

register(
    id='ReacherSpringy-v1',
    entry_point='my_envs.envs:ReacherSpringyEnv',
    max_episode_steps=200,
    reward_threshold=-3.75,
)

register(
    id='ReacherSpringy-v2',
    entry_point='my_envs.envs:ReacherSpringyEnv2',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Sphere-v0',
    entry_point='my_envs.envs:SphereEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='HalfCheetahModified-leg-v12',
    entry_point='my_envs.envs:HalfCheetahEnv_modified_leg',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='InvertedPendulumModified-base-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedBaseEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
    id='InvertedPendulumModified-mass-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedMassEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
    id='InvertedPendulumModified-inertia-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedInertiaEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
id='InvertedPendulumModified-friction-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedFrictionEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
id='InvertedPendulumModified-tilt-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedTiltEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
id='InvertedPendulumModified-motor-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedMotorEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)


register(
id='InvertedPendulumModified-multi-v10',
    entry_point='my_envs.envs:InvertedPendulumModifiedMultiEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
    id='AntModified-base-v11',
    entry_point='my_envs.envs:AntModifiedBaseEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='AntModified-multi-v11',
    entry_point='my_envs.envs:AntModifiedMultiEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HalfCheetahModified-base-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedBaseEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-multi-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedMultiEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-mass-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedMassEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-motor-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedMotorEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-jointfriction-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedJointFrictionEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-groundfriction-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedGroundFrictionEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-damping-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedDampingEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-stiffness-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedStiffnessEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)
