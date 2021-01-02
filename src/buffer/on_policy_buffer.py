import gym
from buffer.buffer import Buffer

class OnPolicyBuffer(Buffer):
    def __init__(self,
                 buffer_size: int,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 device="auto",
                 gae_lambda: float = 1,
                 gamma: float = 0.99):
        super(OnPolicyBuffer, self).__init__(buffer_size,
                               observation_space,
                               action_space,
                               device)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        pass

    def get(self, batch_size: int):
        pass

    def compute_returns_and_advantage(self, last_values, dones):
        pass


if __name__ == '__main__':
    pass