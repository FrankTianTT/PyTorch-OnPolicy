from abc import ABC
import gym

class Buffer(ABC):
    def __init__(self,
                 buffer_size: int,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 device="auto"):
        pass

    def get(self, batch_size: int):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    pass