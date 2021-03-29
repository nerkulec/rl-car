from gym.envs.registration import register

register(
    id='RLCar-v0',
    entry_point='rl_car.envs:RLCar',
)