import pressureplate
import gym
import numpy as np

"""
Up = 0
    Down = 1
    Left = 2
    Right = 3
    Noop = 4
"""

def get_positions(obs):
    positions = []

    for ob in obs:
        positions.append(ob[-2:])

    return np.array(positions)


def get_distance_to_pressureplate(positions, p_plate_pos):
    return np.linalg.norm(p_plate_pos - positions, axis=1, ord=1)


def calc_rewards(positions, p_plate_pos):
    dists = get_distance_to_pressureplate(positions, p_plate_pos)

    return - dists / 10


def test_rewards_correct():
    env = gym.make('pressureplate-linear-2p-v0')
    obs, _ = env.reset()

    pressureplate_pos = np.array([7, 5])
    positions = get_positions(obs)

    for _ in range(1000):
        rand_action = np.random.randint(5)
        obs_, rewards, dones, _ = env.step([4, rand_action])

        my_rewards = calc_rewards(positions, pressureplate_pos)

        assert my_rewards[0] == rewards[0]


def test_rewards_correct2():
    env = gym.make('pressureplate-linear-2p-v0')
    obs, _ = env.reset()

    pressureplate_pos = np.array([7, 5])
    positions = get_positions(obs)

    obs_, rewards, dones, _ = env.step([4, 4])

    my_rewards = calc_rewards(positions, pressureplate_pos)

    assert my_rewards[0] == rewards[0]
