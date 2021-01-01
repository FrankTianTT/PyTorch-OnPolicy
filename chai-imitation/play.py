import gym
import os
from stable_baselines3 import PPO

load_path = os.path.join(os.path.dirname(__file__), 'CartPole')
model = PPO.load(load_path)

env = gym.make("CartPole-v1")


if __name__ == '__main__':
    sum_rewards = 0
    sum_steps = 0

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        sum_rewards += reward
        sum_steps += 1
        env.render()
        if done:
            obs = env.reset()
            print("rewards of {} steps are {}.".format(sum_steps, sum_rewards))
            sum_rewards = 0
            sum_steps = 0