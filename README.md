# To install:
- pull this repo
- (inside folder with setup.py) python -m pip install -e .
- use like this:
    ```python
    import sys
    sys.path.append('path to rl_car')
    import gym
    env = gym.make('rl_car:RLCar-v0')
    env.color = (255, 0, 0) # draw red trajectories
    env.render('trajectories')
    ...
    ```
- you can set different color for training and testing