from on_policy.a2c import A2C
import gym

env = gym.make("Humanoid-v3")
model = A2C(env)

if __name__ == "__main__":
    model.learn(1000000)

    # obs = env.reset()
    # while True:
    #     action = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()