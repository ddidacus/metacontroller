#!/usr/bin/env python3
"""
Evaluate success rate of a trained transformer + metacontroller on BabyAI-MiniBossLevel.

Success = terminated (task completed) OR reward != 0.
"""

from datetime import datetime, timezone
from fire import Fire
from pathlib import Path
from functools import partial
import json

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

# Patch x_transformers so checkpoint config can unpickle (Identity was removed/moved in some versions)
try:
    import x_transformers.x_transformers as _xt_module
    if not hasattr(_xt_module, "Identity"):
        _xt_module.Identity = torch.nn.Identity
except Exception:
    pass

from babyai_env import create_env, get_missions_embeddings
from metacontroller.metacontroller import Transformer, MetaController, extract_grpo_data
from metacontroller.transformer_with_resnet import TransformerWithResnet

MODALITY_RESNET_RGB = "resnet_rgb"
MODALITY_RAW_RGB = "raw_rgb"
MODALITY_SYMBOLIC = "symbolic"

ENV_NAME = "BabyAI-MiniBossLevel-v0"


def exists(v):
    return v is not None


def run_single_rollout(
    unwrapped_model,
    unwrapped_meta_controller,
    env,
    modality: str,
    device: str,
    max_timesteps: int,
    condition_on_mission_embed: bool = False,
    seed: int | None = None,
) -> tuple[bool, int]:
    """
    Run one rollout. Returns (success, episode_length).
    success = terminated OR reward != 0.
    """
    state, _ = env.reset(seed=seed)

    mission = env.unwrapped.mission
    mission_embed = None
    if condition_on_mission_embed:
        mission_embed = get_missions_embeddings([mission]).to(device)

    cache = None
    past_action_id = None

    for step in range(max_timesteps):
        image_tensor = torch.from_numpy(state["image"]).float().unsqueeze(0).to(device)

        if modality == MODALITY_RESNET_RGB:
            image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
            image_tensor = (
                image_tensor - torch.tensor([0.485, 0.456, 0.406], device=device)
            ) / torch.tensor([0.229, 0.224, 0.225], device=device)
            image_tensor = rearrange(image_tensor, "b h w c -> b 1 h w c")
        elif modality == MODALITY_RAW_RGB:
            image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
            image_tensor = (
                image_tensor - torch.tensor([0.485, 0.456, 0.406], device=device)
            ) / torch.tensor([0.229, 0.224, 0.225], device=device)
            image_tensor = rearrange(image_tensor, "b h w c -> b 1 (h w c)")
        elif modality == MODALITY_SYMBOLIC:
            image_tensor = rearrange(image_tensor, "b h w c -> b 1 (h w c)")

        if torch.is_tensor(past_action_id):
            past_action_id = past_action_id.long()

        with torch.no_grad():
            logits, cache = unwrapped_model(
                state=image_tensor,
                actions=past_action_id,
                meta_controller=unwrapped_meta_controller,
                return_cache=True,
                return_raw_action_dist=True,
                cache=cache,
                condition=mission_embed if condition_on_mission_embed else None,
            )

        action = unwrapped_model.action_readout.sample(logits)
        past_action_id = action

        action_id = action.item()
        next_state, reward, terminated, truncated, _ = env.step(action_id)

        if terminated:
            return True, step + 1
        if truncated:
            return (reward != 0), step + 1

        state = next_state

    return False, max_timesteps


def main(
    checkpoint_path: str,
    modality: str = MODALITY_RAW_RGB,
    condition_on_mission_embed: bool = False,
    env_name: str = ENV_NAME,
    seeds: str | list[int] = "11,13,14,19,23,25,26,27,30,32",
    episodes_per_seed: int = 10,
    max_timesteps: int = 100,
    device: str | None = None,
    output_dir: str = "experiments",
):
    """
    Evaluate success rate on unseen seeds.

    Args:
        checkpoint_path: Directory containing model.pt and meta_controller.pt (or
            BabyAI-MiniBossLevel-v0-metacontroller.pt for RL-trained meta).
        modality: raw_rgb, resnet_rgb, or symbolic.
        condition_on_mission_embed: Whether the model conditions on mission embedding.
        env_name: BabyAI environment ID.
        seeds: Comma-separated seeds or list. Unseen seeds for evaluation.
        episodes_per_seed: Number of rollouts per seed.
        max_timesteps: Max steps per episode.
        device: cuda/cpu. Auto if None.
        output_dir: Directory to save JSON results. Default: experiments.
    """
    ckpt_dir = Path(checkpoint_path)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Parse seeds
    if isinstance(seeds, str):
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    else:
        seed_list = list(seeds)

    # Resolve model and meta paths
    model_path = ckpt_dir / "model.pt"
    meta_path_rl = ckpt_dir / f"{env_name}-metacontroller.pt"
    meta_path_bc = ckpt_dir / "meta_controller.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Transformer weights not found: {model_path}")

    meta_path = meta_path_rl if meta_path_rl.exists() else meta_path_bc
    if not meta_path.exists():
        raise FileNotFoundError(
            f"MetaController weights not found at {meta_path_rl} or {meta_path_bc}"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    transformer_klass = (
        TransformerWithResnet if modality == MODALITY_RESNET_RGB else Transformer
    )
    model = transformer_klass.init_and_load(str(model_path), strict=False)
    meta_controller = MetaController.init_and_load(str(meta_path), strict=False)
    model.to(device).eval()
    meta_controller.to(device).eval()

    env_make_fn = partial(
        create_env,
        env_name,
        render_mode="rgb_array",
        use_symbolic=(modality == MODALITY_SYMBOLIC),
    )

    # Run rollouts: for each seed, run episodes_per_seed episodes
    results = {}  # seed -> list of (success, length)
    total_successes = 0
    total_rollouts = 0

    for seed in seed_list:
        results[seed] = []
        env = env_make_fn()

        for ep in tqdm(
            range(episodes_per_seed),
            desc=f"Seed {seed}",
            leave=False,
        ):
            success, length = run_single_rollout(
                model,
                meta_controller,
                env,
                modality=modality,
                device=device,
                max_timesteps=max_timesteps,
                condition_on_mission_embed=condition_on_mission_embed,
                seed=seed,
            )
            results[seed].append((success, length))
            total_successes += int(success)
            total_rollouts += 1

        env.close()

    # Report
    print("\n" + "=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Modality: {modality} | condition_on_mission: {condition_on_mission_embed}")
    print(f"Seeds: {seed_list} | Episodes per seed: {episodes_per_seed}")
    print("=" * 60)

    for seed in seed_list:
        successes = sum(1 for s, _ in results[seed] if s)
        rate = successes / episodes_per_seed
        print(f"  Seed {seed:3d}: {successes}/{episodes_per_seed} = {rate:.2%}")

    overall_rate = total_successes / total_rollouts
    print("-" * 60)
    print(f"  OVERALL: {total_successes}/{total_rollouts} = {overall_rate:.2%}")
    print("=" * 60)

    out = {
        "checkpoint_path": str(ckpt_dir),
        "modality": modality,
        "condition_on_mission_embed": condition_on_mission_embed,
        "env_name": env_name,
        "seeds": seed_list,
        "episodes_per_seed": episodes_per_seed,
        "max_timesteps": max_timesteps,
        "success_rate": overall_rate,
        "total_successes": total_successes,
        "total_rollouts": total_rollouts,
        "per_seed": {
            str(seed): sum(1 for s, _ in results[seed] if s) / episodes_per_seed
            for seed in seed_list
        },
        "eval_time": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ckpt_dir.name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out


if __name__ == "__main__":
    Fire(main)
