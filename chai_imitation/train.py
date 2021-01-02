import pathlib
import pickle
import tempfile
import os
import stable_baselines3 as sb3

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

data_path = os.path.join(os.path.dirname(__file__), 'expert_models', 'cartpole_0', 'rollouts', 'final.pkl')

with open(data_path, "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=2)

log_path = os.path.join(os.path.dirname(__file__), 'log')
save_path = os.path.join(os.path.dirname(__file__), 'CartPole')
logger.configure(log_path)

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
)

if __name__ == '__main__':
    gail_trainer.train(total_timesteps=int(10e4))
    gail_trainer.gen_algo.save(save_path)
