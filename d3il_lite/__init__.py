from gymasium.envs.registration import register

register(
    id="aligning-v0",
    entry_point="aligning:AligningEnv",
    max_episode_steps=400,
    kwargs={'render':False, 'if_vision':False}
)

register(
    id="avoiding-v0",
    entry_point="avoiding:AvoidingEnv",
    max_episode_steps=150,
)

register(
    id="gate_insertion-v0",
    entry_point="inserting:InsertingEnv",
    max_episode_steps=2500,
)

register(
    id="pushing-v0",
    entry_point="pushing:PushingEnv",
    max_episode_steps=500,
)

register(
    id="sorting-v0",
    entry_point="sorting:SortingEnv",
    max_episode_steps=2000,
    kwargs={'max_steps_per_episode': 100, 'render':True, 'num_boxes':2, 'if_vision':False}
)

register(
    id="stacking-v0",
    entry_point="stacking:StackingEnv",
    max_episode_steps=2000,
    kwargs={'max_steps_per_episode': 1000, 'render':True, 'if_vision':False}
)