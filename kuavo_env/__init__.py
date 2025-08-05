from gymnasium.envs.registration import register
register(
    id='Kuavo-Sim',
    entry_point='env.kuavo_sim_env.KuavoSimEnv:KuavoSimEnv',
    max_episode_steps=150,
)

register(
    id='Kuavo-Real',
    entry_point='env.kuavo_real_env.Kuavo-Real:Kuavo-Real',
    max_episode_steps=150,
)