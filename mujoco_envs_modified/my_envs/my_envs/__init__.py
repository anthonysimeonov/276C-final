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
    id='HalfCheetahModified-leg-v0',
    entry_point='my_envs.envs:HalfCheetahEnv_modified_leg',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='InvertedPendulumModified-mass-v0',
    entry_point='my_envs.envs:InvertedPendulumModifiedMassEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
    id='InvertedPendulumModified-inertia-v0',
    entry_point='my_envs.envs:InvertedPendulumModifiedInertia',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
id='InvertedPendulumModified-friction-v0',
    entry_point='my_envs.envs:InvertedPendulumModifiedFrictionEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)

register(
id='InvertedPendulumModified-tilt-v0',
    entry_point='my_envs.envs:InvertedPendulumModifiedTiltEnv',
    max_episode_steps=1000,
    reward_threshold=950,
)
