import os
import sys
SAC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(SAC_PATH)

from on_policy.a2c import A2C
import gym

env = gym.make("LunarLanderContinuous-v2")
model = A2C(env, log_path='log', log_freq=50)

if __name__ == "__main__":
    model.learn(1000000)

    sum_rewards = 0
    sum_steps = 0

    obs = env.reset()
    while True:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
        sum_rewards += reward
        sum_steps += 1
        env.render()
        if done:
            obs = env.reset()
            print("rewards of {} steps are {}.".format(sum_steps, sum_rewards))
            sum_rewards = 0
            sum_steps = 0