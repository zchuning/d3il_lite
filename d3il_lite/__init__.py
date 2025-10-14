from gymnasium.envs.registration import register

register(
    id="aligning-v0",
    entry_point="d3il_lite.envs.aligning:AligningEnv",
    max_episode_steps=400,
    kwargs={"render": False, "if_vision": False},
)

register(
    id="avoiding-v0",
    entry_point="d3il_lite.envs.avoiding:AvoidingEnv",
    max_episode_steps=150,
)

register(
    id="inserting-v0",
    entry_point="d3il_lite.envs.inserting:InsertingEnv",
    max_episode_steps=2500,
)

register(
    id="pushing-v0",
    entry_point="d3il_lite.envs.pushing:PushingEnv",
    max_episode_steps=500,
)

register(
    id="sorting-v0",
    entry_point="d3il_lite.envs.sorting:SortingEnv",
    max_episode_steps=2000,
    kwargs={
        "max_steps_per_episode": 100,
        "render": True,
        "num_boxes": 2,
        "if_vision": False,
    },
)

register(
    id="stacking-v0",
    entry_point="d3il_lite.envs.stacking:StackingEnv",
    max_episode_steps=2000,
    kwargs={"max_steps_per_episode": 1000, "render": True, "if_vision": False},
)
