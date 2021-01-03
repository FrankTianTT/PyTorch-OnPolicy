import os
import sys
SAC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(SAC_PATH)

from on_policy.a2c import A2C
import gym

env = gym.make("LunarLanderContinuous-v2")
model = A2C(env, log_path='log', log_freq=500)

if __name__ == "__main__":
    model.learn(1000000)

    # obs = env.reset()
    # while True:
    #     action = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()