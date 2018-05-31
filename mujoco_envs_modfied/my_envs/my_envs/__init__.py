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
    id='InvertedPendulumModified-Mass',
    entry_point='my_envs.envs:InvertedPendulumModifiedMassEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)
