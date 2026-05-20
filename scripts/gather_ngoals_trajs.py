
# /// script
# dependencies = [
#   "gymnasium",
#   "tqdm",
#   "fire",
#   "memmap-replay-buffer>=0.0.29",
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
from minigrid.wrappers import SymbolicObsWrapper, RGBImgPartialObsWrapper

from memmap_replay_buffer import ReplayBuffer

# helpers

def make_env(env_id, **kwargs):
    if env_id == "NGoals":
        from environments.ngoals import NGoalsEnv
        return NGoalsEnv(**kwargs)
    else:
        return gym.make(env_id, **kwargs)

def exists(val):
    return val is not None

def sample(prob):
    return random.random() < prob

# agent


# functions

def get_bot_cardinal_action(bot, env, last_action):
    from minigrid.core.actions import Actions
    for _ in range(10):
        try:
            action = bot.replan(last_action)
        except Exception:
            return None, last_action
        if action == Actions.left:
            env.unwrapped.agent_dir = (env.unwrapped.agent_dir - 1) % 4
        elif action == Actions.right:
            env.unwrapped.agent_dir = (env.unwrapped.agent_dir + 1) % 4
        elif action == Actions.forward:
            return env.unwrapped.agent_dir + 1, action
        last_action = action
    return None, last_action

def collect_single_episode(env_id, seed, num_steps, random_action_prob, state_shape, use_rgb_states=False, env_kwargs=None):
    from minigrid.core.actions import Actions

    if env_id not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    env = make_env(env_id, render_mode="rgb_array", **(env_kwargs or {}))
    if use_rgb_states:
        env = RGBImgPartialObsWrapper(env)
    else:
        env = SymbolicObsWrapper(env)

    num_actions = env.unwrapped.action_space.n
    state_obs, _ = env.reset(seed=seed)
    episode_state = np.zeros((num_steps, *state_shape), dtype=np.uint8)
    episode_action = np.zeros(num_steps, dtype=np.uint8)
    episode_goal_ids = np.zeros(num_steps, dtype=np.uint8)

    bot = BabyAIBot(env.unwrapped)
    last_action = None
    cumulative_reward = 0

    for step_idx in range(num_steps):
        if random_action_prob > 0 and sample(random_action_prob):
            cardinal = random.randint(1, num_actions)
            last_action = Actions.forward
        else:
            cardinal, last_action = get_bot_cardinal_action(bot, env, last_action)
            if cardinal is None:
                env.close()
                return None, None, None, False, 0, seed

        episode_state[step_idx] = state_obs["image"].copy()
        episode_action[step_idx] = cardinal
        episode_goal_ids[step_idx] = env.unwrapped.get_next_goal_id()
        state_obs, reward, terminated, truncated, info = env.step(cardinal)
        cumulative_reward += reward

        if terminated:
            env.close()
            if cumulative_reward <= 0.0:
                return None, None, None, False, 0, seed
            return episode_state, episode_action, episode_goal_ids, True, step_idx + 1, seed

    env.close()
    return episode_state, episode_action, episode_goal_ids, False, num_steps, seed

def collect_episode_batch(env_id, seeds, num_steps, random_action_prob, state_shape, use_rgb_states=False, env_kwargs=None):
    results = []
    for seed in seeds:
        results.append(collect_single_episode(env_id, seed, num_steps, random_action_prob, state_shape, use_rgb_states, env_kwargs))
    return results

def collect_demonstrations(
    use_rgb_states = False,
    env_id = "NGoals",
    num_trajectories = 100,
    num_pairs = 3,
    num_walls = 4,
    grid_size = 7,
    num_steps = 100,
    random_action_prob = 0.05,
    num_workers = None,
    output_dir = "ngoals-trajectories",
):
    import json
    from environments.ngoals import generate_omitted_pairs

    omitted_pairs = generate_omitted_pairs(rng_seed=0)
    env_kwargs = dict(num_pairs=num_pairs, num_walls=num_walls, size=grid_size, max_steps=num_steps, omitted_pairs=omitted_pairs, mode="train")

    if env_id not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    temp_env = make_env(env_id, **env_kwargs)
    if use_rgb_states:
        temp_env = RGBImgPartialObsWrapper(temp_env)
    else:
        temp_env = SymbolicObsWrapper(temp_env)
    state_shape = temp_env.observation_space['image'].shape
    temp_env.close()

    # Save omitted pairs to output directory
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    omitted_path = output_folder / "omitted_pairs.json"
    with open(omitted_path, "w") as f:
        json.dump(omitted_pairs, f, indent=2)
    logger.info(f"Saved omitted pairs to {omitted_path}")

    logger.info(f"State shape: {state_shape}, env: {env_id}, num_pairs: {num_pairs}, grid_size: {grid_size}, num_walls: {num_walls}")

    if not exists(num_workers):
        num_workers = multiprocessing.cpu_count()

    successful = 0
    successful_seeds = []
    progressbar = tqdm(total=num_trajectories)

    fields = {
        'state': ('uint8', state_shape),
        'action': ('uint8', ()),
        'episode_goal_ids': ('uint8', ()),
    }

    buffer = ReplayBuffer(
        folder = output_folder,
        max_episodes = num_trajectories,
        max_timesteps = num_steps,
        fields = fields,
        overwrite = True,
    )

    batch_size = 64
    max_pending = num_workers * 2
    next_seed = 0

    def make_next_batch():
        nonlocal next_seed
        batch = list(range(next_seed, next_seed + batch_size))
        next_seed += batch_size
        return batch

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for _ in range(min(max_pending, max_pending)):
            batch = make_next_batch()
            future = executor.submit(collect_episode_batch, env_id, batch, num_steps, random_action_prob, state_shape, use_rgb_states, env_kwargs)
            futures[future] = batch

        while futures and successful < num_trajectories:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                batch = futures.pop(future)
                results = future.result()

                for episode_state, episode_action, episode_goal_ids, success, episode_length, returned_seed in results:
                    if successful >= num_trajectories:
                        break
                    if success and exists(episode_state):
                        buffer.store_episode(
                            state = episode_state[:episode_length],
                            action = episode_action[:episode_length],
                            episode_goal_ids = episode_goal_ids[:episode_length],
                        )
                        successful += 1
                        successful_seeds.append(returned_seed)
                        progressbar.update(1)
                        progressbar.set_description(f"seeds tried = {next_seed}")

                if successful < num_trajectories:
                    batch = make_next_batch()
                    new_future = executor.submit(collect_episode_batch, env_id, batch, num_steps, random_action_prob, state_shape, use_rgb_states, env_kwargs)
                    futures[new_future] = batch

    buffer.flush()
    progressbar.close()

    seeds_array = np.array(successful_seeds)
    seeds_path = output_folder / "seeds.npy"
    np.save(seeds_path, seeds_array)

    logger.info(f"Saved {successful} trajectories to {output_dir} (tried {next_seed} seeds)")
    logger.info(f"Saved {len(seeds_array)} successful seeds to {seeds_path}")

if __name__ == "__main__":
    fire.Fire(collect_demonstrations)