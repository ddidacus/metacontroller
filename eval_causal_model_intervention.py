# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch>=0.1.0",
#   "torch-einops-utils>=0.0.27",
#   "minigrid",
#   "tqdm",
#   "sentence-transformers",
#   "imageio"
# ]
# ///

from fire import Fire
from pathlib import Path
from tqdm import tqdm
import random

import json
import numpy as np
import torch
import cv2

from einops import rearrange

from metacontroller.metacontroller import Transformer, extract_grpo_data
from metacontroller.metacontroller_teacher_enforce import EnforcedMetaController
from metacontroller.transformer_with_resnet import TransformerWithResnet

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

def symbolic_to_rgb(symbolic_grid, normalized=False):
    """
    Converts a symbolic grid (H, W, 2+) to an RGB image (H, W, 3).
    """
    # 1. Recover raw indices if normalized
    if normalized:
        obj_max = max(OBJECT_TO_IDX.values())
        col_max = max(COLOR_TO_IDX.values())
        
        grid = symbolic_grid.copy().astype(np.float32)
        grid[:, :, 0] *= obj_max
        grid[:, :, 1] *= col_max
        grid = np.round(grid).astype(np.int32)
    else:
        grid = symbolic_grid.astype(np.int32)

    h, w = grid.shape[:2]
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Matches COLOR_TO_IDX: red=0, green=1, blue=2, purple=3, yellow=4, grey=5, orange=6, cyan=7, pink=8
    color_map = {
        0: [255, 50, 50],    # red
        1: [50, 205, 50],    # green
        2: [50, 100, 255],   # blue
        3: [160, 32, 240],   # purple
        4: [255, 220, 0],    # yellow
        5: [128, 128, 128],  # grey (walls)
        6: [255, 140, 0],    # orange
        7: [0, 220, 220],    # cyan
        8: [255, 105, 180],  # pink
    }

    # Retrieve indices for special handling
    EMPTY_IDX = OBJECT_TO_IDX.get("empty", 1)
    UNSEEN_IDX = OBJECT_TO_IDX.get("unseen", 0)
    AGENT_IDX = OBJECT_TO_IDX.get("agent", 10)

    for y in range(h):
        for x in range(w):
            obj_type = grid[y, x, 0]
            color_idx = grid[y, x, 1]

            # Logic Change: Only color if it's NOT empty, NOT unseen, and NOT null (-1)
            if obj_type > EMPTY_IDX:
                # If it's the agent, make it stand out (White)
                if obj_type == AGENT_IDX:
                    rgb_img[y, x] = [255, 255, 255]
                else:
                    # Use the color dimension for objects like walls, keys, etc.
                    rgb_img[y, x] = color_map.get(color_idx, [200, 200, 200])
            
            elif obj_type == EMPTY_IDX or obj_type == UNSEEN_IDX or obj_type <= 0:
                # Background color for floor/empty space
                rgb_img[y, x] = [25, 25, 25] # Dark charcoal grey

    return rgb_img

def store_trajectory_gif(path:str, trajectory:torch.Tensor, upsample:int=8, fps:float=20.0, frame_skip:int=1):
    import imageio
    from PIL import Image
    rgb_frames = []
    for i, frame in enumerate(trajectory):
        if i % frame_skip != 0:
            continue
        rgb_frame = symbolic_to_rgb(frame.cpu().numpy(), normalized=False)  # H, W, C
        h, w = rgb_frame.shape[:2]
        img = Image.fromarray(rgb_frame, mode='RGB')
        img = img.resize((w * upsample, h * upsample), Image.NEAREST)
        rgb_frames.append(np.array(img))
    imageio.mimsave(path, rgb_frames, fps=fps, loop=0)


# RGB colors for each goal, keyed by TARGET_TO_ID values
GOAL_COLORS_RGB = {
    1: (255, 220, 0),    # yellow
    2: (255, 140, 0),    # orange
    3: (255, 50, 50),    # red
    4: (255, 105, 180),  # pink
    5: (160, 32, 240),   # purple
    6: (0, 220, 220),    # cyan
    7: (50, 100, 255),   # blue
    8: (50, 205, 50),    # green
}

# Normalized [0,1] versions for matplotlib
GOAL_COLORS_MPL = {k: (r/255, g/255, b/255) for k, (r, g, b) in GOAL_COLORS_RGB.items()}


def store_trajectory_gif_with_beta(
    path: str,
    trajectory: torch.Tensor,
    goal_ids: list[int],
    upsample: int = 8,
    fps: float = 20.0,
    frame_skip: int = 1,
):
    """
    Side-by-side GIF: game frame (left) + rolling switch-beta line plot (right).
    The switch beta is a binary spike (1 at goal change, 0 otherwise).
    The line color matches the current goal.
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    T = len(goal_ids)
    # Derive switch beta: 1 wherever the goal changes
    switch_beta = [0] * T
    for t in range(1, T):
        if goal_ids[t] != goal_ids[t - 1]:
            switch_beta[t] = 1

    # Pre-compute game frames – ensure minimum 480px height
    min_height = 480
    sample_frame = trajectory[0].cpu().numpy()
    raw_h, raw_w = sample_frame.shape[:2]
    effective_upsample = max(upsample, int(np.ceil(min_height / raw_h)))

    game_frames = []
    for i, frame in enumerate(trajectory):
        rgb_frame = symbolic_to_rgb(frame.cpu().numpy(), normalized=False)
        h, w = rgb_frame.shape[:2]
        img = Image.fromarray(rgb_frame, mode="RGB")
        img = img.resize((w * effective_upsample, h * effective_upsample), Image.NEAREST)
        game_frames.append(np.array(img))

    game_h, game_w = game_frames[0].shape[:2]
    plot_w, plot_h = game_w, game_h  # match dimensions

    composite_frames = []
    for t in range(T):
        if t % frame_skip != 0:
            continue

        # --- matplotlib plot ---
        plot_dpi = 150
        fig, ax = plt.subplots(figsize=(plot_w / plot_dpi, plot_h / plot_dpi), dpi=plot_dpi)
        fig.patch.set_facecolor("#191919")
        ax.set_facecolor("#191919")

        # Draw colored line segments up to current timestep
        xs = list(range(t + 1))
        for s in range(len(xs) - 1):
            seg_color = GOAL_COLORS_MPL.get(goal_ids[s + 1], (1, 1, 1))
            ax.plot(
                [xs[s], xs[s + 1]],
                [switch_beta[s], switch_beta[s + 1]],
                color=seg_color,
                linewidth=2,
            )

        # Mark switch-beta peaks
        for s in range(t + 1):
            if switch_beta[s] == 1:
                peak_color = GOAL_COLORS_MPL.get(goal_ids[s], (1, 1, 1))
                ax.plot(s, 1, "o", color=peak_color, markersize=6, zorder=5)

        ax.set_xlim(0, T)
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])
        ax.set_ylabel("switch β", color="white", fontsize=9)
        ax.set_xlabel("timestep", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#555555")

        fig.tight_layout(pad=0.4)

        # Rasterize to numpy array
        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        # Resize plot to match game frame height
        plot_pil = Image.fromarray(plot_img).resize((plot_w, game_h), Image.LANCZOS)
        plot_arr = np.array(plot_pil)

        # Composite side-by-side
        composite = np.concatenate([game_frames[t], plot_arr], axis=1)
        composite_frames.append(composite)

    # Append 2 seconds of the final frame as a pause
    num_pause_frames = int(fps * 2)
    composite_frames.extend([composite_frames[-1]] * num_pause_frames)

    imageio.mimsave(path, composite_frames, fps=fps, loop=0)

def binary_betas_to_goal_index(betas:torch.Tensor):
        """
            Convert from binary betas (B, T) to dense goal index (B, T)
            example: a = [0 0 1 0 0 1 0 0 1 0 0]
                    b = [0 0 1 1 1 2 2 2 3 3 3]
            Every timestep carries the current subgoal label (cumsum of betas).
        """
        return torch.cumsum(betas, dim=1).long()

# Patch x_transformers so checkpoint config can unpickle
try:
    import x_transformers.x_transformers as _xt_module
    if not hasattr(_xt_module, "Identity"):
        _xt_module.Identity = torch.nn.Identity
except Exception:
    pass

MODALITY_RESNET_RGB = "resnet_rgb"
MODALITY_RAW_RGB = "raw_rgb"
MODALITY_SYMBOLIC = "symbolic"


def exists(v):
    return v is not None


def create_env(seed, task_length, **kwargs):
    from environments.ngoals import NGoalsEnv
    from minigrid.wrappers import SymbolicObsWrapper
    env = NGoalsEnv(task_length=task_length, training_mode="eval", seed=seed, **kwargs)
    env = SymbolicObsWrapper(env)
    return env


def main(
    transformer_weights_path: str,
    meta_controller_weights_path: str | None = None,
    num_episodes: int = 100,
    max_timesteps: int = 500,
    modality: str = MODALITY_SYMBOLIC,
    dim: int = 512,
    num_goals: int = 8,
    seed: int = 456,
    condition_on_mission_embed: bool = False,
    device: str | None = None,
    output_dir: str | None = None,
):

    assert exists(output_dir), "Insert valid output path"

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    task_lengths = range(2, 8)

    for task_length in task_lengths:
        print(f"======= Seed {seed} =======")
        seed = random.randint(0, 100)
        torch.manual_seed(seed)

        # load transformer
        weights_path = Path(transformer_weights_path)
        assert weights_path.exists(), f"transformer weights not found at {weights_path}"
        transformer_klass = TransformerWithResnet if modality == MODALITY_RESNET_RGB else Transformer
        model = transformer_klass.init_and_load(str(weights_path), strict=False)
        model.eval()
        model.to(device)

        # load metacontroller
        meta_controller = EnforcedMetaController(
            num_goals=num_goals - 1,
            embed_dim=dim,
        )

        if exists(meta_controller_weights_path):
            mc_path = Path(meta_controller_weights_path)
            assert mc_path.exists(), f"meta controller weights not found at {mc_path}"
            state_dict = torch.load(str(mc_path), map_location='cpu', weights_only=False)['model']
            meta_controller.load_state_dict(state_dict, strict=False)
            print(f"Loaded meta controller weights from {mc_path}")

        meta_controller.eval()
        meta_controller.to(device)
        model.meta_controller = meta_controller

        # mission embedding setup
        if condition_on_mission_embed:
            from babyai_env import get_missions_embeddings

        # create single env
        env = create_env(seed=seed, task_length=task_length)

        total_success = 0
        total_episodes = 0

        pbar = tqdm(range(num_episodes), desc='eval')
        print(f"evaluation started, number of goals: {env.unwrapped.task_targets}")

        for ep in pbar:
            ep_seed = torch.randint(0, 1_000_000, (1,)).item()
            state, _ = env.reset()

            mission_embed = None
            if condition_on_mission_embed:
                mission_embed = get_missions_embeddings([env.unwrapped.mission]).to(device)

            cache = None
            past_action_id = None

            episode_goal_ids = []
            episode_states = []
            episode_actions = []
            episode_frames = []

            # print("========\nepisode starts")
            for step in range(max_timesteps):
                
                #  concatenate state, goal_id

                image_tensor = torch.from_numpy(state['image']).float().unsqueeze(0) # (1, H, W, C)
                episode_states.append(image_tensor.flatten(start_dim=1))  # (T, H*W*C)
                episode_goal_ids.append(torch.tensor(env.unwrapped.get_next_goal_id()).unsqueeze(0)) # (1, 1)
                episode_frames.append(torch.from_numpy(state['image']).float().unsqueeze(0)) # T, H, W, C

                # print(f"goal: {env.unwrapped.get_next_goal_id()}")
                
                # prepare tensors

                input_ep_states = torch.cat(episode_states, dim=0).unsqueeze(0).to(device) # (B, T, H*W*C)
                input_ep_actions = None if len(episode_actions) == 0 else torch.cat(episode_actions, dim=0).unsqueeze(0).to(device) # (B, T, )
                input_ep_goal_ids = torch.cat(episode_goal_ids, dim=0).unsqueeze(0).to(device)

                # process trajectory

                with torch.no_grad():
                    # print(input_ep_states.shape)
                    logits, cache = model(
                        state=input_ep_states,
                        actions=past_action_id,
                        meta_controller=meta_controller,
                        return_cache=True,
                        return_raw_action_dist=True,
                        cache=cache,
                        goal_signals=input_ep_goal_ids,
                    )
                    # losses, meta_controller_output = model(
                    #     state=input_ep_states,
                    #     actions=input_ep_actions,
                    #     discovery_phase = True,
                    #     force_behavior_cloning = False,
                    #     return_meta_controller_output = True,
                    #     goal_signals = input_ep_goal_ids
                    # )

                # sample action and store

                # print(logits)
                action = model.action_readout.sample(logits)
                episode_actions.append(torch.tensor(action.item()).unsqueeze(0))
                # print(f"action: {action}")

                # env step

                action_id = action.item()
                next_state, reward, terminated, truncated, _ = env.step(action_id)

                if terminated or truncated:
                    if terminated:
                        episode_frames = torch.cat(episode_frames, dim=0) # T, H, W, C
                        if reward > 0.0:
                            total_success += 1
                            # record successful trajectory
                            print(f"Success, trajectory len: {episode_frames.shape[0]} ; reward: {reward}")
                            store_trajectory_gif(output_dir + f"/tl_{task_length}_success_{total_success}.gif", episode_frames)
                            store_trajectory_gif_with_beta(
                                output_dir + f"/tl_{task_length}_success_{total_success}_beta.gif",
                                episode_frames,
                                goal_ids=[g.item() for g in episode_goal_ids],
                            )
                        # else:
                        #     print(f"Failure, trajectory len: {episode_frames.shape[0]} ; reward: {reward}")
                    break

                state = next_state

            # print(f"successes: {total_success}")
            # print(f"states ({input_ep_states.shape}):\n{input_ep_states.cpu()}")
            # print(f"actions ({input_ep_actions.shape}):\n{input_ep_actions.cpu()}")
            # print(f"goals ({input_ep_goal_ids.shape}):\n{input_ep_goal_ids.cpu()}")
            # exit()

            total_episodes += 1
            pbar.set_postfix(success_rate=f'{total_success / total_episodes:.4f}', successes=f'{total_success}')

        env.close()

        success_rate = total_success / total_episodes
        print(f"\nResults: {total_success}/{total_episodes} episodes succeeded")
        print(f"Success rate: {success_rate:.4f}")

        results = dict(
            success_rate=success_rate,
            total_success=total_success,
            total_episodes=total_episodes,
            transformer_weights_path=transformer_weights_path,
            meta_controller_weights_path=meta_controller_weights_path,
            num_episodes=num_episodes,
            max_timesteps=max_timesteps,
            modality=modality,
            dim=dim,
            num_goals=num_goals,
            seed=seed,
            condition_on_mission_embed=condition_on_mission_embed,
        )

        output_json = output_dir + f"/tl_{task_length}_eval.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_json}")

    return results


if __name__ == '__main__':
    Fire(main)
