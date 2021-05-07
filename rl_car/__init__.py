from gym.envs.registration import register

register(
    id='RLCar-v0',
    entry_point='rl_car.envs:RLCarV0',
)
register(
    id='RLCar-v1',
    entry_point='rl_car.envs:RLCarV1',
)
register(
    id='RLCar-v2',
    entry_point='rl_car.envs:RLCarV2',
)