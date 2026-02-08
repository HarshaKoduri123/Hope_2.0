# rl/chunk_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sentence_transformers import SentenceTransformer
import torch
import re


def safe_split(text):
    sentences = re.split(r'(?<=\.|\?)\s+', text)
    if len(sentences) <= 1:
        return text, ""
    mid = len(sentences) // 2
    return " ".join(sentences[:mid]), " ".join(sentences[mid:])


class ChunkEnv(gym.Env):
    """
    RL environment for chunk boundary optimization using HOPE.
    """

    def __init__(self, document, initial_chunks, hope_metric, eval_every=5):
        super().__init__()

        self.document = document
        self.chunks = list(initial_chunks)
        self.hope = hope_metric
        self.eval_every = eval_every

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )

        self.chunk_emb_cache = {}
        self.current_idx = 0
        self.prev_hope = None
        self.step_count = 0

        self.action_space = spaces.Discrete(3) 

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(384 + 2,),
            dtype=np.float32
        )

    # --------------------------------------------------

    def _embed_chunk(self, text):
        if text not in self.chunk_emb_cache:
            self.chunk_emb_cache[text] = self.encoder.encode(
                text, normalize_embeddings=True
            )
        return self.chunk_emb_cache[text]

    def _get_obs(self):
        emb = self._embed_chunk(self.chunks[self.current_idx])

        pos_ratio = self.current_idx / max(1, len(self.chunks) - 1)
        len_ratio = len(self.chunks[self.current_idx]) / 1000.0

        return np.concatenate([emb, [pos_ratio, len_ratio]]).astype(np.float32)

    # --------------------------------------------------

    def reset(self, seed=None):
        self.current_idx = 0
        self.step_count = 0
        self.prev_hope = None
        self.chunk_emb_cache.clear()
        return self._get_obs(), {}

    # --------------------------------------------------

    def step(self, action):
        self.step_count += 1

        # ---- Apply action ----
        if action == 0 and self.current_idx < len(self.chunks) - 1:
            self.chunks[self.current_idx] += " " + self.chunks.pop(self.current_idx + 1)

        elif action == 2:
            left, right = safe_split(self.chunks[self.current_idx])
            if right.strip():
                self.chunks[self.current_idx] = left
                self.chunks.insert(self.current_idx + 1, right)

        self.current_idx += 1
        done = self.current_idx >= len(self.chunks)

        reward = 0.0

        # ---- Evaluate HOPE sparsely or at end ----
        if done or self.step_count % self.eval_every == 0:
            metrics = self.hope.calculate_hope(
                self.document,
                self.chunks
            )

            hope_score = metrics["hope_score"]

            if self.prev_hope is not None:
                reward += hope_score - self.prev_hope

            self.prev_hope = hope_score

        # ---- Regularization ----
        reward -= 0.005 * len(self.chunks)

        return (
            self._get_obs() if not done else self._get_obs(),
            reward,
            done,
            False,
            {}
        )
