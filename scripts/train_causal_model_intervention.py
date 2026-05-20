# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "metacontroller-pytorch>=0.1.0",
#   "torch",
#   "einops",
#   "tqdm",
#   "wandb",
#   "gymnasium",
#   "minigrid",
#   "sentence-transformers",
#   "matplotlib"
# ]
# ///

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import random
import re
import glob as glob_mod
import signal
import fire
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.nn import init
import torch.nn as nn
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange
from torch_einops_utils import maybe, lens_to_mask

import matplotlib.pyplot as plt
import wandb

from metacontroller import EnforcedMetaController
from metacontroller import MetaController, Transformer, binary_entropy
from metacontroller.transformer_with_resnet import TransformerWithResnet

from torch.nn.parallel import DistributedDataParallel

import minigrid
import gymnasium as gym

MODALITY_RESNET_RGB = "resnet_rgb"
MODALITY_RAW_RGB = "raw_rgb"
MODALITY_SYMBOLIC = "symbolic"


# --- on-the-fly trajectory collection (adapted from scripts/gather_ngoals_trajs.py) ---

def _get_bot_cardinal_action(bot, env, last_action):
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


def collect_trajectory(env, seed, num_steps, state_shape):
    from minigrid.utils.baby_ai_bot import BabyAIBot

    state_obs = env.reset(seed=seed)[0]

    episode_state = np.zeros((num_steps, *state_shape), dtype=np.uint8)
    episode_action = np.zeros(num_steps, dtype=np.uint8)
    episode_goal_ids = np.zeros(num_steps, dtype=np.uint8)

    bot = BabyAIBot(env.unwrapped)
    last_action = None
    cumulative_reward = 0

    for step_idx in range(num_steps):
        cardinal, last_action = _get_bot_cardinal_action(bot, env, last_action)
        if cardinal is None:
            return None

        episode_state[step_idx] = state_obs["image"]
        episode_action[step_idx] = cardinal
        episode_goal_ids[step_idx] = env.unwrapped.get_next_goal_id()
        state_obs, reward, terminated, truncated, _ = env.step(cardinal)
        cumulative_reward += reward

        if terminated:
            if cumulative_reward <= 0.0:
                return None
            length = step_idx + 1
            return dict(
                state=episode_state[:length],
                action=episode_action[:length],
                episode_goal_ids=episode_goal_ids[:length],
                length=length,
            )

    return None


def _collect_one_successful(env_config, env_mode, num_steps, state_shape, max_seed):
    """Worker function: creates its own env, retries until a successful trajectory."""
    from minigrid.wrappers import SymbolicObsWrapper
    from environments.ngoals import NGoalsEnv

    env_inner = NGoalsEnv(seed=1, config=env_config, mode=env_mode)
    env = SymbolicObsWrapper(env_inner)

    while True:
        seed = random.randint(1, max_seed)
        traj = collect_trajectory(env, seed, num_steps, state_shape)
        if traj is not None:
            env.close()
            return traj


def collect_batch(env_config, env_mode, batch_size, num_steps, state_shape, max_seed, num_workers=1):
    from concurrent.futures import ProcessPoolExecutor

    args = (env_config, env_mode, num_steps, state_shape, max_seed)

    if num_workers <= 1:
        trajectories = [_collect_one_successful(*args) for _ in range(batch_size)]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_collect_one_successful, *args) for _ in range(batch_size)]
            trajectories = [f.result() for f in futures]

    max_len = max(t["length"] for t in trajectories)

    batch_states = np.zeros((batch_size, max_len, *state_shape), dtype=np.uint8)
    batch_actions = np.zeros((batch_size, max_len), dtype=np.uint8)
    batch_goal_ids = np.zeros((batch_size, max_len), dtype=np.uint8)
    batch_lens = np.zeros(batch_size, dtype=np.int64)

    for i, traj in enumerate(trajectories):
        L = traj["length"]
        batch_states[i, :L] = traj["state"]
        batch_actions[i, :L] = traj["action"]
        batch_goal_ids[i, :L] = traj["episode_goal_ids"]
        batch_lens[i] = L

    return {
        "state": torch.from_numpy(batch_states),
        "action": torch.from_numpy(batch_actions).long(),
        "episode_goal_ids": torch.from_numpy(batch_goal_ids).long(),
        "_lens": torch.from_numpy(batch_lens),
    }


# --- helpers ---

def set_requires_grad(network: nn.Module, grad_val: bool):
    for param in network.parameters():
        param.requires_grad = grad_val

def initialize_weights_xavier(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def visualize_switch_betas(
    switch_betas,      # (B, T-1)
    episode_lens,      # (B,) or None
    gradient_step,
    num_samples = 3,
    accelerator = None
):
    B, T_minus_1 = switch_betas.shape

    num_samples = min(num_samples, B)
    sample_indices = np.random.choice(B, size=num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    fig.suptitle(f'Step {gradient_step} | Switch Betas Visualization', fontsize=10)

    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(sample_indices):
        if episode_lens is not None:
            ep_len = int(episode_lens[idx].item())
        else:
            ep_len = T_minus_1

        sample_switch_betas = switch_betas[idx, :ep_len-1].detach().cpu()

        ax = axes[i]
        ax.plot(sample_switch_betas.numpy(), label='switch betas', linewidth=2)
        ax.set_xlabel('timesteps')
        ax.set_ylabel(f'switch betas (sample {idx})')
        ax.legend(loc='upper right')

    plt.tight_layout()

    if exists(accelerator):
        tracker = accelerator.get_tracker("wandb")
        if exists(tracker):
            tracker.log({
                f"switch_betas/step_{gradient_step}": wandb.Image(fig)
            }, step=gradient_step)

    plt.close(fig)


def _sample_solvable_eval_seeds(env_config, num_seeds, traj_length, state_shape, max_seed):
    """Sample random seeds that are solvable by the bot (test mode)."""
    from minigrid.wrappers import SymbolicObsWrapper
    from environments.ngoals import NGoalsEnv

    env_inner = NGoalsEnv(seed=1, config=env_config, mode="test")
    env = SymbolicObsWrapper(env_inner)

    seeds = []
    while len(seeds) < num_seeds:
        seed = random.randint(1, max_seed)
        traj = collect_trajectory(env, seed, traj_length, state_shape)
        if traj is not None:
            seeds.append(seed)

    env.close()
    return seeds


def evaluate_ngoals_success_rate(
    model,
    meta_controller,
    accelerator,
    gradient_step,
    env_config,
    traj_length,
    state_shape,
    max_seed,
    num_eval = 8,
    max_timesteps = 500,
    use_wandb = True,
):
    from minigrid.wrappers import SymbolicObsWrapper
    from environments.ngoals import NGoalsEnv, TARGET_TO_ID

    eval_seeds = _sample_solvable_eval_seeds(env_config, num_eval, traj_length, state_shape, max_seed)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_mc = accelerator.unwrap_model(meta_controller)
    unwrapped_model.eval()
    unwrapped_mc.eval()

    device = next(unwrapped_model.parameters()).device
    total_success = 0
    subgoals_completed_per_episode = []

    env_inner = NGoalsEnv(seed=1, config=env_config, render_mode="rgb_array", mode="test")
    total_goals_in_sequence = len(env_inner.task_targets)
    env = SymbolicObsWrapper(env_inner)

    for seed in eval_seeds:
        state = env.reset(seed=seed)[0]

        cache = None
        past_action_id = None
        episode_states = []
        episode_goal_ids = []

        for step in range(max_timesteps):

            image_tensor = torch.from_numpy(state['image']).float().unsqueeze(0)
            episode_states.append(image_tensor.flatten(start_dim=1))
            episode_goal_ids.append(torch.tensor(env.unwrapped.get_next_goal_id()).unsqueeze(0))

            input_ep_states = torch.cat(episode_states, dim=0).unsqueeze(0).to(device)
            input_ep_goal_ids = torch.cat(episode_goal_ids, dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, cache = unwrapped_model(
                    state=input_ep_states,
                    actions=past_action_id,
                    meta_controller=unwrapped_mc,
                    return_cache=True,
                    return_raw_action_dist=True,
                    cache=cache,
                    goal_signals=input_ep_goal_ids,
                )

            logits[..., 0] = float('-inf')
            action = unwrapped_model.action_readout.sample(logits)
            past_action_id = action

            action_id = action.item()
            next_state, reward, terminated, truncated, _ = env.step(action_id)

            if terminated or truncated:
                if terminated and reward > 0:
                    total_success += 1
                break

            state = next_state

        subgoals_completed_per_episode.append(env.unwrapped.next_goal_idx)

    env.close()

    success_rate = total_success / len(eval_seeds)
    mean_subgoals = np.mean(subgoals_completed_per_episode)
    max_subgoals = np.max(subgoals_completed_per_episode)
    mean_subgoal_rate = mean_subgoals / total_goals_in_sequence

    unwrapped_model.train()
    unwrapped_mc.train()

    if use_wandb and accelerator.is_main_process:
        accelerator.log({
            "eval/ngoals_success_rate": success_rate,
            "eval/ngoals_successes": total_success,
            "eval/ngoals_total": len(eval_seeds),
            "eval/ngoals_mean_subgoals_completed": mean_subgoals,
            "eval/ngoals_max_subgoals_completed": max_subgoals,
            "eval/ngoals_mean_subgoal_completion_rate": mean_subgoal_rate,
            "eval/ngoals_total_goals_in_sequence": total_goals_in_sequence,
        }, step=gradient_step)

    accelerator.print(
        f"[Eval step {gradient_step}] NGoals success rate: {success_rate:.4f} ({total_success}/{len(eval_seeds)}) | "
        f"subgoals: mean={mean_subgoals:.2f}, max={max_subgoals}/{total_goals_in_sequence}, rate={mean_subgoal_rate:.4f}"
    )

    return success_rate


def train(
    run_seed = 42,
    num_trajectories = 20_000_000,
    env_config = "configs/environment/metacontroller_simplified.yml",
    modality = MODALITY_SYMBOLIC,
    batch_size = 1024,
    num_workers = 1,
    gradient_accumulation_steps = None,
    lr = 3e-4,
    discovery_lr = 3e-4,
    lr_schedule = "constant",
    weight_decay = 0.03,
    discovery_weight_decay = 0.03,
    dim = 512,
    depth = 8,
    heads = 8,
    dim_head = 64,
    switch_temperature = 1.,
    use_wandb = False,
    wandb_project = "metacontroller",
    checkpoint_dir = ".",
    load_transformer_weights_path = None,
    load_meta_controller_weights_path = None,
    save_steps = 1000,
    eval_steps = 1000,
    state_loss_weight = 1e-3,
    action_loss_weight = 1.,
    discovery_obs_loss_weight = 1e-3,
    discovery_action_recon_loss_weight = 1.,
    discovery_entropy_loss_weight = 0.0,
    discovery_ratio_loss_weight = 0.0,
    normalize_state_action_losses = False,
    max_grad_norm = 1.,
    visual_recon_loss_weight = 1.0,
    lr_min_ratio = 0.05,
):

    torch.manual_seed(run_seed)

    steps_per_phase = num_trajectories // batch_size

    # environment setup for on-the-fly trajectory collection

    from minigrid.wrappers import SymbolicObsWrapper
    from environments.ngoals import NGoalsEnv, load_env_config

    env_cfg = load_env_config(env_config)
    num_goals = env_cfg["num_colors"]

    temp_env_inner = NGoalsEnv(seed=1, config=env_config, mode="train")
    temp_env = SymbolicObsWrapper(temp_env_inner)
    state_shape = temp_env.observation_space['image'].shape
    num_actions = int(temp_env_inner.action_space.n) + 1
    traj_length = temp_env_inner.max_steps
    temp_env.close()

    if modality == MODALITY_RESNET_RGB:
        state_dim = 256
    elif modality == MODALITY_RAW_RGB or modality == MODALITY_SYMBOLIC:
        state_dim = int(torch.tensor(state_shape).prod().item())

    # accelerator

    accelerator = Accelerator(
        log_with = "wandb" if use_wandb else None,
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters = True)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator.print(f"Using device: {device} (accelerator.device: {accelerator.device})")

    accelerator.print(f"Detected state_dim: {state_dim}, num_actions: {num_actions}, traj_length: {traj_length}")
    accelerator.print(f"Steps per phase: {steps_per_phase} (num_trajectories={num_trajectories}, batch_size={batch_size})")

    # try to recover wandb run id from the latest checkpoint for seamless resume
    wandb_run_id = None
    if use_wandb:
        existing_ckpts = sorted(glob_mod.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt")))
        if existing_ckpts:
            step_numbers = []
            for ckpt_path in existing_ckpts:
                m = re.search(r'checkpoint_step_(\d+)\.pt$', ckpt_path)
                if m:
                    step_numbers.append((int(m.group(1)), ckpt_path))
            if step_numbers:
                step_numbers.sort()
                _, latest_ckpt_path = step_numbers[-1]
                ckpt_meta = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
                wandb_run_id = ckpt_meta.get("wandb_run_id")
                if wandb_run_id:
                    accelerator.print(f"Found wandb run id in checkpoint: {wandb_run_id}")

    if use_wandb:
        wandb_config = {
            "run_seed": run_seed,
            "num_trajectories": num_trajectories,
            "env_config": env_config,
            "batch_size": batch_size,
            "lr": lr,
            "lr_schedule": lr_schedule,
            "lr_min_ratio": lr_min_ratio if lr_schedule == "cosine" else None,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "dim_head": dim_head,
            "state_loss_weight": state_loss_weight,
            "action_loss_weight": action_loss_weight,
            "discovery_ratio_loss_weight": discovery_ratio_loss_weight,
            "switch_temperature": switch_temperature,
        }

        if wandb_run_id and accelerator.is_main_process:
            def _timeout_handler(signum, frame):
                raise TimeoutError("wandb resume timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            try:
                signal.alarm(30)
                wandb.init(project=wandb_project, id=wandb_run_id, resume="must", config=wandb_config)
                signal.alarm(0)
                accelerator.print(f"Resumed wandb run: {wandb_run_id}")
            except Exception as e:
                signal.alarm(0)
                accelerator.print(f"Failed to resume wandb run {wandb_run_id} ({e}), starting new run")
                wandb_run_id = None
                wandb.init(project=wandb_project, config=wandb_config)
            finally:
                signal.signal(signal.SIGALRM, old_handler)

            wandb_run_id = wandb.run.id
            tracker = accelerator.get_tracker("wandb", unwrap=True)
            tracker.run = wandb.run
        else:
            accelerator.init_trackers(wandb_project, config=wandb_config)
            if accelerator.is_main_process:
                wandb_run_id = wandb.run.id
                accelerator.print(f"New wandb run: {wandb_run_id}")

    def store_checkpoint(step: int = None, is_discovering: bool = False):
        if accelerator.is_main_process:
            ckpt_dir = checkpoint_dir
            if exists(step):
                ckpt_file = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            else:
                ckpt_file = os.path.join(ckpt_dir, "checkpoint_final.pt")

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_mc = accelerator.unwrap_model(meta_controller)

            ckpt = {
                "model_state_dict": unwrapped_model.state_dict(),
                "meta_controller_state_dict": unwrapped_mc.state_dict(),
                "optim_model_state_dict": optim_model.state_dict(),
                "optim_meta_controller_state_dict": optim_meta_controller.state_dict(),
                "scheduler_model_state_dict": scheduler_model.state_dict() if scheduler_model is not None else None,
                "gradient_step": step if exists(step) else 0,
                "is_discovering": is_discovering,
                "wandb_run_id": wandb_run_id,
                "config": {
                    "run_seed": run_seed,
                    "num_trajectories": num_trajectories,
                    "env_config": env_config,
                    "batch_size": batch_size,
                    "lr": lr,
                    "discovery_lr": discovery_lr,
                    "lr_schedule": lr_schedule,
                    "lr_min_ratio": lr_min_ratio,
                    "weight_decay": weight_decay,
                    "discovery_weight_decay": discovery_weight_decay,
                    "dim": dim,
                    "depth": depth,
                    "heads": heads,
                    "dim_head": dim_head,
                    "modality": modality,
                    "state_loss_weight": state_loss_weight,
                    "action_loss_weight": action_loss_weight,
                    "discovery_obs_loss_weight": discovery_obs_loss_weight,
                    "discovery_action_recon_loss_weight": discovery_action_recon_loss_weight,
                    "num_goals": num_goals,
                },
            }
            torch.save(ckpt, ckpt_file)
            accelerator.print(f"Checkpoint saved to {ckpt_file} (step={step})")

    # meta controller (teacher-enforced)
    meta_controller = EnforcedMetaController(
        num_goals = num_goals,
        embed_dim = dim
    )

    # transformer

    transformer_class = TransformerWithResnet if modality == MODALITY_RESNET_RGB else Transformer

    transformer_kwargs = dict(
        dim = dim,
        state_embed_readout = dict(
            num_continuous = state_dim,
            readout_kwargs = dict(continuous_log_var_embed = True)
        ),
        action_embed_readout = dict(
            num_discrete = num_actions
        ),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        meta_controller = meta_controller,
        normalize_state_action_losses = normalize_state_action_losses
    )
    if modality == MODALITY_RESNET_RGB:
        transformer_kwargs["use_layernorm"] = True

    model = transformer_class(**transformer_kwargs)

    # load pre-trained weights for fine-tuning if given / otherwise xavier init

    if exists(load_transformer_weights_path):
        _path = Path(load_transformer_weights_path)
        assert _path.exists(), f"load_transformer_weights_path {load_transformer_weights_path} does not exist"
        if hasattr(model, "load"):
            model.load(str(_path))
        else:
            state = torch.load(_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded transformer weights from {load_transformer_weights_path}")
    else:
        for name, child in model.named_children():
            if name != "meta_controller":
                child.apply(initialize_weights_xavier)
        accelerator.print("Initialized transformer with Xavier")

    if exists(load_meta_controller_weights_path):
        _path = Path(load_meta_controller_weights_path)
        assert _path.exists(), f"load_meta_controller_weights_path {load_meta_controller_weights_path} does not exist"
        if hasattr(meta_controller, "load"):
            meta_controller.load(str(_path))
        else:
            state = torch.load(_path, map_location="cpu", weights_only=True)
            meta_controller.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded meta_controller weights from {load_meta_controller_weights_path}")
    else:
        meta_controller.apply(initialize_weights_xavier)
        accelerator.print("Initialized meta_controller with Xavier")

    # optimizer

    optim_model = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)

    optim_meta_controller = AdamW(meta_controller.discovery_parameters(), lr = discovery_lr, weight_decay = discovery_weight_decay)

    # prepare (no dataloader — batches are collected on-the-fly)

    model, optim_model, optim_meta_controller = accelerator.prepare(model, optim_model, optim_meta_controller)
    model = model.to(device)
    meta_controller = meta_controller.to(device)

    # optional LR schedule for BC phase
    scheduler_model = None
    if lr_schedule == "cosine":
        scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_model, T_max=steps_per_phase, eta_min=lr * lr_min_ratio
        )
        accelerator.print(f"Using cosine LR schedule for BC: {steps_per_phase} steps, eta_min={lr * lr_min_ratio:.2e}")

    # auto-resume from latest checkpoint

    resume_step = 0
    existing_ckpts = sorted(glob_mod.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt")))
    if existing_ckpts:
        step_numbers = []
        for ckpt in existing_ckpts:
            m = re.search(r'checkpoint_step_(\d+)\.pt$', ckpt)
            if m:
                step_numbers.append((int(m.group(1)), ckpt))
        if step_numbers:
            step_numbers.sort()
            resume_step, latest_ckpt_path = step_numbers[-1]

            accelerator.print(f"Auto-resuming from checkpoint: {latest_ckpt_path}")
            ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)

            accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
            accelerator.unwrap_model(meta_controller).load_state_dict(ckpt["meta_controller_state_dict"])
            optim_model.load_state_dict(ckpt["optim_model_state_dict"])
            optim_meta_controller.load_state_dict(ckpt["optim_meta_controller_state_dict"])

            if scheduler_model is not None and ckpt.get("scheduler_model_state_dict") is not None:
                scheduler_model.load_state_dict(ckpt["scheduler_model_state_dict"])
                accelerator.print(f"Restored LR scheduler state")

            accelerator.print(f"Resumed: step={resume_step}, is_discovering={ckpt.get('is_discovering', False)}")

    # compute phase boundaries

    skip_bc = load_transformer_weights_path is not None
    bc_phase_steps = 0 if skip_bc else steps_per_phase
    discovery_phase_start = bc_phase_steps
    total_steps = bc_phase_steps + steps_per_phase

    def get_phase(step):
        if step < discovery_phase_start:
            return "behavior_cloning"
        else:
            return "discovery_phase"

    if resume_step > 0:
        accelerator.print(f"Will skip to step {resume_step} / {total_steps} (phase: {get_phase(resume_step)})")

    # training

    old_discovery_obs_loss_weight = discovery_obs_loss_weight
    old_discovery_action_recon_loss_weight = discovery_action_recon_loss_weight

    prev_phase = None
    is_behavior_cloning = False
    is_discovering = False
    discovery_step = max(0, resume_step - discovery_phase_start) if resume_step > discovery_phase_start else 0

    total_losses = defaultdict(float)
    log_count = 0

    progress_bar = tqdm(range(total_steps), desc="Training", disable=not accelerator.is_local_main_process)

    for gradient_step in progress_bar:

        if gradient_step < resume_step:
            continue

        current_phase = get_phase(gradient_step)

        # phase transition: set requires_grad and checkpoint at boundary
        if current_phase != prev_phase:

            if current_phase == "behavior_cloning":
                is_behavior_cloning = True
                is_discovering = False

                if isinstance(model, DistributedDataParallel): set_requires_grad(model.module, True)
                else: set_requires_grad(model, True)

                if isinstance(meta_controller, DistributedDataParallel): set_requires_grad(meta_controller.module, False)
                else: set_requires_grad(meta_controller, False)

            elif current_phase == "discovery_phase":
                if prev_phase is not None:
                    accelerator.wait_for_everyone()
                    store_checkpoint(step=gradient_step, is_discovering=False)

                is_behavior_cloning = False
                is_discovering = True

                if isinstance(model, DistributedDataParallel): set_requires_grad(model.module, False)
                else: set_requires_grad(model, False)

                if isinstance(meta_controller, DistributedDataParallel): set_requires_grad(meta_controller.module, True)
                else: set_requires_grad(meta_controller, True)

                if isinstance(model, DistributedDataParallel):
                    model.module.meta_controller_reset_kl_loss_warmup()
                else:
                    model.meta_controller_reset_kl_loss_warmup()

            prev_phase = current_phase
            total_losses = defaultdict(float)
            log_count = 0
            progress_bar.set_description(f"Training [{current_phase}]")

        optim = optim_model if not is_discovering else optim_meta_controller

        # collect batch on-the-fly
        batch = collect_batch(env_config, "train", batch_size, traj_length, state_shape, max_seed=num_trajectories, num_workers=num_workers)
        batch = {k: v.to(device) for k, v in batch.items()}

        goal_signals = batch['episode_goal_ids'].long()

        if modality == MODALITY_RESNET_RGB:
            states = batch['state'].float()
            states = torch.clamp(states / 255.0, min=0.0, max=1.0)
            states = (states - torch.tensor([0.485, 0.456, 0.406]).to(states.device)) / torch.tensor([0.229, 0.224, 0.225]).to(states.device)
        elif modality == MODALITY_RAW_RGB:
            states = batch['state'].float()
            states = torch.clamp(states / 255.0, min=0.0, max=1.0)
            states = (states - torch.tensor([0.485, 0.456, 0.406]).to(states.device)) / torch.tensor([0.229, 0.224, 0.225]).to(states.device)
            states = rearrange(states, 'b t ... -> b t (...)')
        elif modality == MODALITY_SYMBOLIC:
            states = batch['state'].float()
            states = rearrange(states, 'b t ... -> b t (...)')

        actions = batch['action'].long()
        episode_lens = batch.get('_lens')

        mission_embeddings = None

        with accelerator.accumulate(model):
            visual_autoencoder_loss = None

            if modality == MODALITY_RESNET_RGB:
                (losses, meta_controller_output), visual_autoencoder_loss = model(
                    state=states,
                    actions=actions,
                    episode_lens = episode_lens,
                    discovery_phase = is_discovering,
                    force_behavior_cloning = not is_discovering,
                    return_meta_controller_output = True,
                    condition = mission_embeddings,
                    return_visual_autoencoder_loss = True,
                    goal_signals = goal_signals
                )
            else:
                losses, meta_controller_output = model(
                    state=states,
                    actions=actions,
                    episode_lens = episode_lens,
                    discovery_phase = is_discovering,
                    force_behavior_cloning = not is_discovering,
                    return_meta_controller_output = True,
                    condition = mission_embeddings,
                    goal_signals = goal_signals
                )

            if is_behavior_cloning:
                state_loss, action_loss = losses
                loss = (
                    state_loss * state_loss_weight +
                    action_loss * action_loss_weight
                )
                log = dict(
                    state_loss = state_loss.item(),
                    action_loss = action_loss.item(),
                )

            elif is_discovering:
                obs_loss, action_recon_loss, kl_loss, ratio_loss = losses

                switch_mask = maybe(lens_to_mask)(episode_lens, meta_controller_output.switch_beta.shape[1])
                if exists(switch_mask):
                    n_valid = switch_mask.sum()
                    entropy_loss = (binary_entropy(meta_controller_output.switch_beta) * switch_mask).sum() / n_valid.item()
                    last_hard_switch_density = ((meta_controller_output.switch_beta > 0.5).float() * switch_mask).sum().item() / n_valid.item()
                    last_soft_switch_density = (meta_controller_output.switch_beta * switch_mask).sum().item() / n_valid.item()
                else:
                    entropy_loss = binary_entropy(meta_controller_output.switch_beta).mean()
                    last_hard_switch_density = (meta_controller_output.switch_beta > 0.5).float().mean().item()
                    last_soft_switch_density = meta_controller_output.switch_beta.mean().item()

                entropy_weight = discovery_entropy_loss_weight
                discovery_obs_loss_weight = old_discovery_obs_loss_weight
                discovery_action_recon_loss_weight = old_discovery_action_recon_loss_weight

                loss = (
                    obs_loss * discovery_obs_loss_weight
                    + action_recon_loss * discovery_action_recon_loss_weight
                    + entropy_loss * entropy_weight
                    + kl_loss
                    + ratio_loss
                )

                if exists(visual_autoencoder_loss):
                    loss = loss + visual_autoencoder_loss * visual_recon_loss_weight

                discovery_step += 1

                log = dict(
                    obs_loss = obs_loss.item(),
                    action_loss = action_recon_loss.item(),
                    kl_loss = kl_loss.item(),
                    kl_loss_weight = meta_controller_output.kl_loss_weight,
                    entropy_loss = entropy_loss.item(),
                    ratio_loss = ratio_loss.item(),
                    hard_switch_density = last_hard_switch_density,
                    soft_switch_density = last_soft_switch_density,
                )
                if exists(visual_autoencoder_loss):
                    log["visual_autoencoder_loss"] = visual_autoencoder_loss.item()

            if gradient_accumulation_steps is not None: loss /= gradient_accumulation_steps

            accelerator.backward(loss)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)

            if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
                optim.step()
                optim.zero_grad()

                if not is_discovering and scheduler_model is not None:
                    scheduler_model.step()

                if is_discovering:
                    m = accelerator.unwrap_model(model)
                    m.meta_controller_maybe_increment_kl_loss_step()

        # logging

        if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:

            for key, value in log.items():
                total_losses[key] += value
            log_count += 1

            prefix = "discovery_phase/" if is_discovering else "behavior_cloning/"

            m = accelerator.unwrap_model(model)
            log_dict = {f"{prefix}{k}": v for k, v in log.items()}
            log_dict[f"{prefix}total_loss"] = loss.item()
            log_dict[f"{prefix}grad_norm"] = grad_norm.item()

            if m.running_bc_state_loss is not None:
                log_dict[f"{prefix}running_bc_state_loss"] = m.running_bc_state_loss.squeeze().item()
            if m.running_bc_action_loss is not None:
                log_dict[f"{prefix}running_bc_action_loss"] = m.running_bc_action_loss.squeeze().item()
            if m.running_discovery_state_loss is not None:
                log_dict[f"{prefix}running_discovery_state_loss"] = m.running_discovery_state_loss.squeeze().item()
            if m.running_discovery_action_loss is not None:
                log_dict[f"{prefix}running_discovery_action_loss"] = m.running_discovery_action_loss.squeeze().item()

            if not is_discovering and scheduler_model is not None:
                log_dict[f"{prefix}lr"] = scheduler_model.get_last_lr()[0]

            accelerator.log(log_dict, step=gradient_step)

            progress_bar.set_postfix(**log)

        # checkpoint

        if (gradient_step + 1) % save_steps == 0:
            accelerator.wait_for_everyone()
            store_checkpoint(step=gradient_step + 1, is_discovering=is_discovering)

        if (gradient_step + 1) % eval_steps == 0 and is_behavior_cloning and accelerator.is_main_process:
            model.eval()
            with torch.no_grad():
                eval_batch = collect_batch(env_config, "test", min(batch_size, 256), traj_length, state_shape, max_seed=num_trajectories, num_workers=min(num_workers, 8))
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                eval_states = eval_batch['state'].float()
                eval_states = rearrange(eval_states, 'b t ... -> b t (...)')
                eval_actions = eval_batch['action'].long()
                eval_lens = eval_batch.get('_lens')
                eval_goal_signals = eval_batch['episode_goal_ids'].long()

                eval_losses, _ = model(
                    state=eval_states, actions=eval_actions, episode_lens=eval_lens,
                    discovery_phase=False, force_behavior_cloning=True,
                    return_meta_controller_output=True, condition=None,
                    goal_signals=eval_goal_signals,
                )
                eval_state_loss, eval_action_loss = eval_losses

                eval_T = eval_states.shape[1]
                eval_embed = model(state=eval_states, actions=eval_actions, episode_lens=eval_lens,
                                   discovery_phase=False, force_behavior_cloning=True,
                                   return_embed=True, condition=None)
                # can't easily get per-timestep from model forward; log aggregate eval losses
                bc_eval_log = {
                    "behavior_cloning/eval_state_loss": eval_state_loss.item(),
                    "behavior_cloning/eval_action_loss": eval_action_loss.item(),
                }
                accelerator.log(bc_eval_log, step=gradient_step)
                accelerator.print(f"  [BC eval] state_loss={eval_state_loss.item():.4f}, action_loss={eval_action_loss.item():.4f}")
            model.train()

        if (gradient_step + 1) % eval_steps == 0 and is_discovering:
            if use_wandb and accelerator.is_main_process:
                visualize_switch_betas(
                    switch_betas = meta_controller_output.switch_beta,
                    episode_lens = episode_lens,
                    gradient_step = gradient_step + 1,
                    num_samples = 3,
                    accelerator = accelerator
                )

            if accelerator.is_main_process:
                evaluate_ngoals_success_rate(
                    model = model,
                    meta_controller = meta_controller,
                    accelerator = accelerator,
                    gradient_step = gradient_step + 1,
                    env_config = env_config,
                    traj_length = traj_length,
                    state_shape = state_shape,
                    max_seed = num_trajectories,
                    num_eval = 8,
                    max_timesteps = 500,
                    use_wandb = use_wandb,
                )

    # save final weights
    accelerator.wait_for_everyone()
    store_checkpoint(step=total_steps, is_discovering=is_discovering)

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
