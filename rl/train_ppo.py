# rl/train_ppo.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.chunk_env import ChunkEnv

def train_chunk_optimizer(document, passages, hope_metric):
    env = DummyVecEnv([
        lambda: ChunkEnv(document, passages, hope_metric)
    ])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=128,
        gamma=0.95,
        learning_rate=2.5e-4,
        ent_coef=0.005,
        clip_range=0.2,
        verbose=1
    )


    model.learn(total_timesteps=20_000)
    model.save("hope_mrl_chunker")

    return model
