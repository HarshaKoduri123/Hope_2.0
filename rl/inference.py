# rl/inference.py

from rl.chunk_env import ChunkEnv

def optimize_chunks(model, document, passages, hope_metric):
    env = ChunkEnv(document, passages, hope_metric)
    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    return env.chunks
