# /// script
# dependencies = [
#   "gymnasium",
#   "minigrid",
#   "tqdm",
#   "fire",
#   "memmap-replay-buffer>=0.0.23",
#   "loguru"
# ]
# ///

# taken with modifications from https://github.com/ddidacus/bot-minigrid-babyai/blob/main/tests/get_trajectories.py

import fire
import random
import multiprocessing
from loguru import logger
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import torch
import minigrid
import gymnasium as gym
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper

from memmap_replay_buffer import ReplayBuffer

# helpers

def exists(val):
    return val is not None

def sample(prob):
    return random.random() < prob

# agent

class BabyAIBotEpsilonGreedy:
    def __init__(self, env, random_action_prob = 0.):
        self.expert = BabyAIBot(env)
        self.random_action_prob = random_action_prob
        self.num_actions = env.action_space.n
        self.last_action = None

    def __call__(self, state):
        if sample(self.random_action_prob):
            action = torch.randint(0, self.num_actions, ()).item()
        else:
            action = self.expert.replan(self.last_action)

        self.last_action = action
        return action

# functions

def collect_single_episode(env_id, seed, num_steps, random_action_prob, state_shape):
    """
    Collect a single episode of demonstrations.
    Returns tuple of (episode_state, episode_action, success, episode_length)
    """
    if env_id not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    env = gym.make(env_id, render_mode="rgb_array", highlight=False)
    env = FullyObsWrapper(env.unwrapped)
    env = SymbolicObsWrapper(env.unwrapped)

    try:
        state_obs, _ = env.reset(seed=seed)
        
        episode_state = np.zeros((num_steps, *state_shape), dtype=np.float32)
        episode_action = np.zeros(num_steps, dtype=np.float32)

        expert = BabyAIBotEpsilonGreedy(env.unwrapped, random_action_prob = random_action_prob)

        for _step in range(num_steps):
            try:
                action = expert(state_obs)
            except Exception:
                env.close()
                return None, None, False, 0

            episode_state[_step] = state_obs["image"]
            episode_action[_step] = action

            state_obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                env.close()
                return episode_state, episode_action, True, _step + 1

        env.close()
        return episode_state, episode_action, False, num_steps

    except Exception:
        env.close()
        return None, None, False, 0

def collect_demonstrations(
    env_id = "BabyAI-MiniBossLevel-v0",
    num_seeds = 100,
    num_episodes_per_seed = 100,
    num_steps = 500,
    random_action_prob = 0.05,
    num_workers = None,
    output_dir = "babyai-minibosslevel-trajectories"
):
    """
    The BabyAI Bot should be able to solve all BabyAI environments,
    allowing us therefore to generate demonstrations.
    Parallelized version using ProcessPoolExecutor.
    """

    # Register minigrid envs if not already registered
    if env_id not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    # Determine state shape from environment
    temp_env = gym.make(env_id)
    temp_env = FullyObsWrapper(temp_env.unwrapped)
    temp_env = SymbolicObsWrapper(temp_env.unwrapped)
    state_shape = temp_env.observation_space['image'].shape
    temp_env.close()

    logger.info(f"Detected state shape: {state_shape} for env {env_id}")

    if not exists(num_workers):
        num_workers = multiprocessing.cpu_count()

    total_episodes = num_seeds * num_episodes_per_seed

    # Prepare seeds for all episodes
    seeds = []
    for count in range(num_seeds):
        for it in range(num_episodes_per_seed):
            seeds.append(count + 1)

    successful = 0
    progressbar = tqdm(total=total_episodes)

    output_folder = Path(output_dir)

    fields = {
        'state': ('float', state_shape),
        'action': ('float', ())
    }

    buffer = ReplayBuffer(
        folder = output_folder,
        max_episodes = total_episodes,
        max_timesteps = num_steps,
        fields = fields,
        overwrite = True
    )

    # Parallel execution with bounded pending futures to avoid OOM
    max_pending = num_workers * 4

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        seed_iter = iter(seeds)
        futures = {}

        # Initial batch of submissions
        for _ in range(min(max_pending, len(seeds))):
            seed = next(seed_iter, None)
            if exists(seed):
                future = executor.submit(collect_single_episode, env_id, seed, num_steps, random_action_prob, state_shape)
                futures[future] = seed

        # Process completed tasks and submit new ones
        while futures:
            # Wait for at least one future to complete
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                del futures[future]
                episode_state, episode_action, success, episode_length = future.result()

                if success and exists(episode_state):
                    buffer.store_episode(
                        state = episode_state[:episode_length],
                        action = episode_action[:episode_length]
                    )
                    successful += 1

                progressbar.update(1)
                progressbar.set_description(f"success rate = {successful}/{progressbar.n:.2f}")

                # Submit a new task to replace the completed one
                seed = next(seed_iter, None)
                if exists(seed):
                    new_future = executor.submit(collect_single_episode, env_id, seed, num_steps, random_action_prob, state_shape)
                    futures[new_future] = seed

    buffer.flush()
    progressbar.close()

    logger.info(f"Saved {successful} trajectories to {output_dir}")

if __name__ == "__main__":
    fire.Fire(collect_demonstrations)
