from __future__ import annotations
import random
import imageio
import numpy as np
from pathlib import Path
from itertools import product

import yaml

from environments.bot import BabyAIBotEpsilonGreedy
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.spaces import Discrete
from minigrid.envs.babyai.core.verifier import GoToInstr, AndInstr, ObjDesc


TARGETS = [
    "yellow",
    "orange",
    "red",
    "pink",
    "purple",
    "cyan",
    "blue",
    "green",
]

TARGET_TO_ID = {
    "yellow": 1,
    "orange": 2,
    "red": 3,
    "pink": 4,
    "purple": 5,
    "cyan": 6,
    "blue": 7,
    "green": 8,
}

# All 56 ordered pairs of distinct colors (8 x 7)
ALL_PAIRS = [(a, b) for a, b in product(TARGETS, TARGETS) if a != b]


def load_env_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class NGoalsEnv(MiniGridEnv):
    def __init__(
        self,
        seed: int,
        config: str | Path,
        size: int = None,
        num_walls: int = 0,
        max_steps: int | None = None,
        mode: str = "train",
        **kwargs,
    ):
        assert mode in ("train", "test"), f"mode must be 'train' or 'test', got '{mode}'"

        cfg = load_env_config(config)
        self._train_sequences = cfg["train_tasks"]
        self._test_sequences = cfg["test_tasks"]
        if size is None:
            size = cfg.get("grid_size", 8)
        if max_steps is None:
            max_steps = cfg.get("trajectory_length")

        self._seed = seed
        self._num_walls = num_walls
        self._mode = mode
        self.task_targets = None
        self.next_goal_idx = 0

        self.reset_seed()
        self.set_task_targets()

        mission_space = MissionSpace(mission_func=self._gen_mission, ordered_placeholders=[[self]])

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            highlight=False,
            max_steps=max_steps,
            **kwargs,
        )
        self.action_space = Discrete(4, start=1)

    def get_next_goal_id(self):
        return TARGET_TO_ID[self.task_targets[self.next_goal_idx]]

    def reset_seed(self):
        random.seed(self._seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self.reset_seed()
        self.set_task_targets()
        return super().reset(seed=seed, options=options)

    def set_task_targets(self):
        if self._mode == "test":
            seq = random.choice(self._test_sequences)
        else:
            seq = random.choice(self._train_sequences)
        self.task_targets = [TARGETS[i] for i in seq]

    @staticmethod
    def _gen_mission(self):
        return "move to: " + ", ".join(self.task_targets)

    def _gen_grid(self, width, height):
        self.mission = "move to: " + ", ".join(self.task_targets)
        self.next_goal_idx = 0

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        inner_cells = [
            (x, y)
            for x in range(1, width - 1)
            for y in range(1, height - 1)
        ]

        # 1) Place random interior walls
        wall_positions = random.sample(inner_cells, self._num_walls)
        for x, y in wall_positions:
            self.grid.set(x, y, Wall())

        free_cells = [c for c in inner_cells if c not in set(wall_positions)]

        # 2) Place goal tiles for each unique color in the sequence
        unique_colors = list(dict.fromkeys(self.task_targets))
        goal_cells = random.sample(free_cells, len(unique_colors))
        free_cells = [c for c in free_cells if c not in set(goal_cells)]

        goal_descs = {}
        for idx, goal_color in enumerate(unique_colors):
            x, y = goal_cells[idx]
            obj = Goal(color=goal_color)
            self.put_obj(obj, x, y)
            desc = ObjDesc("goal", color=goal_color)
            desc.obj_set = [obj]
            desc.obj_poss = [(x, y)]
            goal_descs[goal_color] = desc

        # 3) Place agent on a remaining free cell
        agent_cell = free_cells[random.randint(0, len(free_cells) - 1)]
        self.agent_pos = agent_cell
        self.agent_dir = random.randint(0, 3)

        # Build instructions for the Bot
        instrs = GoToInstr(goal_descs[self.task_targets[0]])
        for color in self.task_targets[1:]:
            instrs = AndInstr(instrs, GoToInstr(goal_descs[color]))
        self.instrs = instrs

    def step(self, action):
        self.agent_dir = int(action) - 1
        obs, reward, terminated, truncated, info = super().step(Actions.forward)

        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell is not None and agent_cell.type == "goal":
            color = agent_cell.color
            current_target = self.task_targets[self.next_goal_idx]
            if color == current_target:
                self.next_goal_idx += 1
                if self.next_goal_idx >= len(self.task_targets):
                    terminated = True
                    reward = 1.0
                else:
                    terminated = False
                    reward = 0.0
            elif color in self.task_targets[:self.next_goal_idx]:
                terminated = False
                reward = 0.0
            else:
                terminated = False
                reward = -1.0

        return obs, reward, terminated, truncated, info


def main():

    for epoch in range(10):
    
        # Environment initialization
        random_seed = random.randint(0, 100)
        env = NGoalsEnv(
            render_mode="rgb_array", 
            seed=random_seed
        )
        env.reset()
        init_frame = env.render()
        
        # Start expert demonstration
        print(f"Seed: {random_seed}")
        print(env.mission)


        num_steps = 100
        state_obs, _ = env.reset()
        # state_shape = env.observation_space['image'].shape
        state_shape = init_frame.shape
        episode_state = np.zeros((num_steps, *state_shape), dtype=np.uint8)
        episode_action = np.zeros(num_steps, dtype=np.uint8)

        bot = BabyAIBotEpsilonGreedy(env.unwrapped)

        episode_reward = 0
        for _step in range(num_steps):
            # Sample action
            action = bot(state_obs)

            # obs_image = state_obs["image"].copy()
            obs_image = env.render()

            # Log and step
            episode_state[_step] = obs_image
            episode_action[_step] = action
            state_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated:
                print(f"Final reward: {episode_reward}")
                env.close()
                break
        env.close()
    
    
    # # enable manual control for testing
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()

    
if __name__ == "__main__":
    main()