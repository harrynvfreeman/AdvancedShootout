from gym.envs.registration import register

register(
    id='AdvancedShootoutRandom-v0',
    entry_point='gym_advancedshootout.envs:AdvancedShootoutRandomEnv',
)

register(
    id='AdvancedShootoutSmart-v0',
    entry_point='gym_advancedshootout.envs:AdvancedShootoutSmartEnv',
)