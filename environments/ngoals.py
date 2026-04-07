from __future__ import annotations
import random
import imageio
import numpy as np
from itertools import permutations

from environments.bot import BabyAIBotEpsilonGreedy
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.babyai.core.verifier import GoToInstr, AndInstr, ObjDesc


# 6 inner-wall layouts (agent always at (1,1)).
# Each lambda(w, h) returns a list of (x, y) wall cells.
# Goal positions are paired in _GOAL_POSITIONS — 4 (x,y) per layout, placed in the
# distinct navigable regions formed by that layout's walls.
_WALL_LAYOUTS = [
    # 0: Cross — solid cross, no hole in the middle
    lambda w, h: (
        [(5, y) for y in range(2, h - 2)] +
        [(x, 5) for x in range(2, w - 2) if x != 5]
    ),
    # 1: H-barriers — two vertical walls with gaps at mid-height (y=5)
    lambda w, h: (
        [(3, y) for y in range(2, 5)] + [(3, y) for y in range(6, h - 2)] +
        [(7, y) for y in range(2, 5)] + [(7, y) for y in range(6, h - 2)]
    ),
    # 2: Staggered corridors — two horizontal walls with offset gaps (top gap right, bottom gap left)
    lambda w, h: (
        [(x, 3) for x in range(2, w - 2) if x != 7] +
        [(x, 7) for x in range(2, w - 2) if x != 3]
    ),
    # 3: Z-shape — diagonal wall dividing the grid into two connected regions
    lambda w, h: (
        [(x, 3) for x in range(2, 6)] +
        [(5, y) for y in range(3, 8)] +
        [(x, 7) for x in range(5, w - 2)]
    ),
    # 4: L-shape — top wall with a descending left arm, entry gap at top-right
    lambda w, h: (
        [(x, 3) for x in range(2, w - 3)] +
        [(2, y) for y in range(3, h - 2)]
    ),
    # 5: U-shape — mid horizontal wall with two upward arms, center gap
    lambda w, h: (
        [(x, 5) for x in range(2, 5)] + [(x, 5) for x in range(6, w - 2)] +
        [(2, y) for y in range(2, 5)] +
        [(w - 3, y) for y in range(2, 5)]
    ),
]

_GOAL_POSITIONS = [
    # 0: Cross — 8 goals spread across 4 quadrants (2 per quadrant)
    [[2, 2], [4, 2], [7, 2], [8, 4], [2, 7], [4, 8], [7, 7], [8, 8]],
    # 1: H-barriers — spread across 3 corridors (left, middle, right)
    [[1, 2], [2, 8], [4, 2], [6, 8], [5, 5], [4, 5], [8, 2], [9, 8]],
    # 2: Staggered corridors — spread across 3 horizontal bands
    [[2, 1], [8, 2], [2, 5], [8, 5], [4, 4], [6, 6], [2, 9], [8, 8]],
    # 3: Z-shape — spread across both regions
    [[2, 1], [4, 2], [8, 2], [8, 5], [2, 5], [2, 8], [4, 9], [8, 9]],
    # 4: L-shape — scattered in the accessible interior
    [[4, 1], [8, 2], [5, 4], [8, 5], [5, 6], [8, 7], [5, 8], [8, 9]],
    # 5: U-shape — 3 upper (between arms), 5 lower
    [[4, 2], [6, 3], [5, 4], [1, 7], [4, 7], [7, 7], [3, 9], [8, 9]],
]

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

SUBTARGETS = [
    ["yellow", "orange"],
    ["orange", "red"],
    ["red", "pink"],
    ["pink", "purple"],
    ["purple", "cyan"],
    ["cyan", "blue"],
    ["blue", "green"],
]



class NGoalsEnv(MiniGridEnv):
    def __init__(
        self,
        seed=42,
        size=11,
        training_mode="train",
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        task_length=8,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.mode = training_mode
        self._seed = seed
        self._task_length = task_length

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.goals_order = None
        self.goals_achievement = {}
        self.next_goal = None

        self.reset_seed()
        self.set_task_targets()

        mission_space = MissionSpace(mission_func=self._gen_mission, ordered_placeholders=[[self]])

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def get_next_goal_id(self):
        return TARGET_TO_ID[self.next_goal]

    def reset_seed(self):
        random.seed(self._seed)

    def reset(self, *, seed=None, options=None):
        # Use the provided seed, or fall back to the constructor default
        if seed is not None:
            self._seed = seed
            self.reset_seed()
        self.set_task_targets()
        return super().reset(seed=seed, options=options)

    def set_task_targets(self):
        # Select tasks depending on mode
        if self.mode == "train":
            task_idx = random.randint(0, len(SUBTARGETS)-1)
            self.task_targets = SUBTARGETS[task_idx]
        elif self.mode == "eval":
            self.task_targets = TARGETS[:self._task_length]
        else: 
            raise NotImplementedError()

    @staticmethod
    def _gen_mission(self):
        ACTION = "move to: "
        SEP = ", "
        goals_permutations = list(permutations(self.task_targets))
        rand_idx = random.randint(0, len(goals_permutations)-1)
        mission = ACTION + SEP.join(self.task_targets)
        
        # Randomize order on the grid
        self.goals_order = goals_permutations[rand_idx]

        return mission

    def _gen_grid(self, width, height):
        # Regenerate goals_order from current task_targets
        # (_gen_mission only runs during __init__, not on every reset)
        goals_permutations = list(permutations(self.task_targets))
        rand_idx = random.randint(0, len(goals_permutations) - 1)
        self.goals_order = goals_permutations[rand_idx]
        self.mission = "move to: " + ", ".join(self.task_targets)

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Pick layout: varies with seed (random state already seeded in __init__)
        self.layout_idx = random.randint(0, len(_WALL_LAYOUTS) - 1)
        for x, y in _WALL_LAYOUTS[self.layout_idx](width, height):
            self.grid.set(x, y, Wall())

        # Place the agent at a random free cell (not wall, not goal position)
        goal_pos_set = set(tuple(p) for p in _GOAL_POSITIONS[self.layout_idx])
        free_cells = [
            (x, y)
            for x in range(1, width - 1)
            for y in range(1, height - 1)
            if self.grid.get(x, y) is None and (x, y) not in goal_pos_set
        ]
        self.agent_pos = free_cells[random.randint(0, len(free_cells) - 1)]
        self.agent_dir = random.randint(0, 3)

        # Goals to achieve are FIXED, their position is randomized
        self.goals_achievement = {c: False for c in self.task_targets}
        self.next_goal = self.task_targets[0]

        # Place the goals in regions defined by the chosen layout
        positions = _GOAL_POSITIONS[self.layout_idx]
        goal_descs = {}
        for idx, goal_color in enumerate(self.goals_order):
            x, y = positions[idx]
            obj = Goal(color=goal_color)
            self.put_obj(obj, x, y)
            desc = ObjDesc("goal", color=goal_color)
            desc.obj_set = [obj]
            desc.obj_poss = [(x, y)]
            goal_descs[goal_color] = desc

        # Build instructions for the Bot with pre-populated ObjDescs
        instrs = GoToInstr(goal_descs[self.task_targets[0]])
        for color in self.task_targets[1:]:
            instrs = AndInstr(instrs, GoToInstr(goal_descs[color]))
        self.instrs = instrs

    def step(self, action):                                                                                  
        obs, reward, terminated, truncated, info = super().step(action)                                        
                                                                                                            
        # The BabyAI bot navigates *next to* the target and faces it
        # (GoNextToSubgoal pops when fwd_pos == target). Check the cell
        # the agent is facing. This is safe on every step because BFS
        # treats Goal objects as blockers and paths around them — the
        # agent only ends up facing a goal after explicit navigation.
        fwd_pos = self.agent_pos + self.dir_vec
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is not None and fwd_cell.type == "goal":
            color = fwd_cell.color
            if color == self.next_goal:
                self.goals_achievement[color] = True
                terminated = all(self.goals_achievement.values())
                if terminated:
                    reward = 1.0
                else:
                    # reward = 1.0 / float(len(self.goals_order))
                    reward = 0.0
                    self.next_goal = next(
                        (c for c in self.task_targets if not self.goals_achievement[c]), None
                    )
            else:
                # Already visited: tolerate
                if self.goals_achievement[color] == True:
                    reward = 0.0
                # Otherwise game over
                else:
                    terminated = True
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