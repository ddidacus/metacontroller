# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch>=0.1.0",
#   "torch-einops-utils>=0.0.27",
#   "minigrid",
#   "tqdm",
#   "sentence-transformers"
# ]
# ///

from fire import Fire
from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import torch

from einops import rearrange

from metacontroller.metacontroller import Transformer, extract_grpo_data
from metacontroller.metacontroller_teacher_enforce import EnforcedMetaController
from metacontroller.transformer_with_resnet import TransformerWithResnet

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


def create_env(seed, **kwargs):
    from environments.ngoals import NGoalsEnv
    from minigrid.wrappers import SymbolicObsWrapper
    env = NGoalsEnv(task_length=8, training_mode="eval", seed=seed, **kwargs)
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
    output_json: str | None = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    env = create_env(seed=seed)

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

        # print("========\nepisode starts")
        for step in range(max_timesteps):
            
            #  concatenate state, goal_id

            image_tensor = torch.from_numpy(state['image']).float().unsqueeze(0) # (1, H, W, C)
            episode_states.append(image_tensor.flatten(start_dim=1))  # (T, H*W*C)
            episode_goal_ids.append(torch.tensor(env.unwrapped.get_next_goal_id()).unsqueeze(0)) # (1, 1)

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
                if terminated and reward > 0.0:
                    total_success += 1
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

    if exists(output_json):
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


if __name__ == '__main__':
    Fire(main)
