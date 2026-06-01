# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "metacontroller-pytorch>=0.1.0",
#   "torch-einops-utils>=0.0.27",
#   "torch",
#   "einops",
#   "minigrid",
#   "tqdm",
#   "wandb",
#   "matplotlib"
# ]
# ///

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import re
import glob as glob_mod
from fire import Fire
from pathlib import Path
from tqdm import tqdm
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import cat, tensor, Tensor
from torch.optim import Adam

from accelerate import Accelerator

from einops import rearrange
from torch_einops_utils import lens_to_mask

import matplotlib.pyplot as plt
import wandb

from metacontroller.metacontroller import (
    Transformer, MetaController, z_score, extract_grpo_data, binary_entropy
)
from metacontroller.transformer_with_resnet import TransformerWithResnet

from gymnasium import spaces
from gymnasium.core import Wrapper
from gymnasium.vector import AsyncVectorEnv

MODALITY_RESNET_RGB = "resnet_rgb"
MODALITY_RAW_RGB = "raw_rgb"
MODALITY_SYMBOLIC = "symbolic"

try:
    import x_transformers.x_transformers as _xt_module
    if not hasattr(_xt_module, "Identity"):
        _xt_module.Identity = torch.nn.Identity
except Exception:
    pass

CARDINAL_ACTION_NAMES = ['north', 'east', 'south', 'west']


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def set_requires_grad(network: nn.Module, grad_val: bool):
    for param in network.parameters():
        param.requires_grad = grad_val


# environment

class ImageOnlyWrapper(Wrapper):
    """Strip mission/direction from obs — keep only 'image' for vectorized batching."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"image": env.observation_space["image"]})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {"image": obs["image"]}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {"image": obs["image"]}, reward, terminated, truncated, info


def make_ngoals_env(env_config, mode="test", render_mode=None):
    from minigrid.wrappers import SymbolicObsWrapper
    from environments.ngoals import NGoalsEnv
    env = NGoalsEnv(seed=1, config=env_config, mode=mode, render_mode=render_mode)
    env = SymbolicObsWrapper(env)
    env = ImageOnlyWrapper(env)
    return env


# reward shaping

def reward_shaping_fn(
    cumulative_rewards: Tensor,
    all_rewards: Tensor,
    episode_lens: Tensor,
    reject_threshold_cumulative_reward_variance: float = 0.
) -> Tensor | None:
    if exists(reject_threshold_cumulative_reward_variance):
        if cumulative_rewards.var() < reject_threshold_cumulative_reward_variance:
            return None
    return cumulative_rewards

def should_reject_group_based_on_switch_betas(
    switch_betas: Tensor,
    episode_lens: Tensor
):
    return switch_betas.sum().item() == 0.


# visualization

def visualize_eval_trajectory(
    unwrapped_model,
    unwrapped_meta_controller,
    env_config,
    modality: str,
    gradient_step: int,
    max_timesteps: int = 500,
    seed: int = 0,
    use_wandb: bool = False,
    accelerator = None,
    fps: int = 4,
):
    device = accelerator.device if exists(accelerator) else 'cpu'

    env = make_ngoals_env(env_config, mode="test", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)

    mission = env.unwrapped.mission

    cache = None
    past_action_id = None

    frames = []
    actions_list = []
    rewards_list = []
    switch_betas_list = []

    for step in range(max_timesteps):
        rgb_frame = env.render()

        image_tensor = torch.from_numpy(state['image']).float().unsqueeze(0).to(device)

        if modality == MODALITY_RESNET_RGB:
            image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
            image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406], device=device)) / torch.tensor([0.229, 0.224, 0.225], device=device)
            image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')
        elif modality == MODALITY_RAW_RGB:
            image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
            image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406], device=device)) / torch.tensor([0.229, 0.224, 0.225], device=device)
            image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')
        elif modality == MODALITY_SYMBOLIC:
            image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')

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
            )

        logits[..., 0] = float('-inf')
        action = unwrapped_model.action_readout.sample(logits)
        past_action_id = action

        grpo_data = extract_grpo_data(unwrapped_meta_controller, cache)
        switch_beta = grpo_data.switch_beta.item()

        env_action = action.item() - 1
        next_state, reward, terminated, truncated, _ = env.step(env_action)

        action_name = CARDINAL_ACTION_NAMES[env_action] if env_action < len(CARDINAL_ACTION_NAMES) else str(env_action)

        frames.append(rgb_frame)
        actions_list.append(action_name)
        rewards_list.append(reward)
        switch_betas_list.append(switch_beta)

        if terminated or truncated:
            break

        state = next_state

    env.close()

    rendered = []
    dpi = 80
    for i, (frame, act, rew, sb) in enumerate(zip(frames, actions_list, rewards_list, switch_betas_list)):
        fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(7, 3.4), dpi=dpi,
                                               gridspec_kw={'width_ratios': [1, 2]})
        fig.suptitle(f'mission: "{mission}"', fontsize=8, y=0.98)

        ax_img.imshow(frame)
        ax_img.set_title(f"a={act}  r={rew:.2f}  β={sb:.2f}", fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_plot.plot(range(i + 1), switch_betas_list[:i + 1], linewidth=2, color='#1f77b4')
        ax_plot.set_xlim(0, max_timesteps)
        ax_plot.set_ylim(-0.05, 1.05)
        ax_plot.set_xlabel('timesteps')
        ax_plot.set_ylabel('switch betas')

        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        rendered.append(img)
        plt.close(fig)

    video_np = np.stack(rendered)
    video_np = rearrange(video_np, 't h w c -> 1 t c h w')

    if use_wandb and exists(accelerator) and accelerator.is_main_process:
        wandb.log(
            {"grpo/eval/trajectory_video": wandb.Video(video_np, fps=fps, format="mp4")},
            step=gradient_step,
        )


def visualize_switch_betas(
    switch_betas,
    episode_lens,
    gradient_step,
    num_samples = 3,
    use_wandb = False,
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

    if use_wandb and exists(accelerator) and accelerator.is_main_process:
        wandb.log({
            f"grpo/switch_betas/step_{gradient_step}": wandb.Image(fig)
        }, step=gradient_step)

    plt.close(fig)


# main

def main(
    env_config = "configs/environment/metacontroller_paper.yml",
    checkpoint_path: str = None,
    num_episodes = int(10e6),
    max_timesteps = 500,
    batch_size = 16,
    lr = 3e-5,
    kl_loss_weight = 0.01,
    num_epochs = 4,
    max_grad_norm = None,
    modality = MODALITY_SYMBOLIC,
    dim = 512,
    depth = 8,
    heads = 8,
    dim_head = 64,
    save_steps = 100,
    eval_steps = 50,
    output_dir = ".",
    use_wandb = False,
    wandb_project = 'metacontroller',
    reject_threshold_cumulative_reward_variance = None,
    entropy_loss_weight = 0.0,
    entropy_warmup_steps = 0,
):

    torch.manual_seed(456)

    if not exists(max_grad_norm): max_grad_norm = float('inf')

    assert exists(checkpoint_path), "checkpoint_path is required (from train_metacontroller.py)"
    assert batch_size >= 2, "batch_size must be at least 2 for GRPO"

    # resolve checkpoint: if directory, find latest; if file, use directly

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        existing = sorted(glob_mod.glob(str(ckpt_path / "checkpoint_step_*.pt")))
        step_numbers = []
        for p in existing:
            m = re.search(r'checkpoint_step_(\d+)\.pt$', p)
            if m:
                step_numbers.append((int(m.group(1)), p))
        assert step_numbers, f"No checkpoints found in {checkpoint_path}"
        step_numbers.sort()
        _, ckpt_file = step_numbers[-1]
    else:
        ckpt_file = str(ckpt_path)

    print(f"Loading checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    ckpt_config = ckpt.get("config", {})
    dim = ckpt_config.get("dim", dim)
    depth = ckpt_config.get("depth", depth)
    heads = ckpt_config.get("heads", heads)
    dim_head = ckpt_config.get("dim_head", dim_head)
    modality = ckpt_config.get("modality", modality)
    step_offset = ckpt.get("gradient_step", 0)

    # environment

    from minigrid.wrappers import SymbolicObsWrapper
    from environments.ngoals import NGoalsEnv, load_env_config

    temp_env_inner = NGoalsEnv(seed=1, config=env_config, mode="test")
    temp_env = SymbolicObsWrapper(temp_env_inner)
    state_shape = temp_env.observation_space['image'].shape
    num_actions = int(temp_env_inner.action_space.n) + 1
    temp_env.close()

    if modality == MODALITY_RESNET_RGB:
        state_dim = 256
    elif modality == MODALITY_RAW_RGB or modality == MODALITY_SYMBOLIC:
        state_dim = int(torch.tensor(state_shape).prod().item())

    print(f"Env: state_dim={state_dim}, num_actions={num_actions}, state_shape={state_shape}")

    # accelerator

    accelerator = Accelerator(log_with='wandb' if use_wandb else None)

    wandb_run_id = ckpt.get("wandb_run_id")

    if use_wandb:
        grpo_config = {
            "grpo_env_config": env_config,
            "grpo_checkpoint_path": checkpoint_path,
            "grpo_lr": lr,
            "grpo_kl_loss_weight": kl_loss_weight,
            "grpo_batch_size": batch_size,
            "grpo_num_epochs": num_epochs,
            "grpo_entropy_loss_weight": entropy_loss_weight,
            "grpo_entropy_warmup_steps": entropy_warmup_steps,
        }

        if accelerator.is_main_process:
            import signal as _signal

            if wandb_run_id:
                def _timeout_handler(signum, frame):
                    raise TimeoutError("wandb resume timed out")

                old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
                try:
                    _signal.alarm(30)
                    wandb.init(project=wandb_project, id=wandb_run_id, resume="must", config=grpo_config)
                    _signal.alarm(0)
                    accelerator.print(f"Resumed wandb run: {wandb_run_id}")
                except Exception as e:
                    _signal.alarm(0)
                    accelerator.print(f"Failed to resume wandb run {wandb_run_id} ({e}), starting new run")
                    wandb.init(project=wandb_project, config=grpo_config)
                finally:
                    _signal.signal(_signal.SIGALRM, old_handler)
            else:
                wandb.init(project=wandb_project, config=grpo_config)
                accelerator.print(f"New wandb run: {wandb.run.id}")

    # vectorized environment for parallel rollouts (split across GPUs)

    local_batch_size = batch_size // accelerator.num_processes
    assert local_batch_size >= 2, f"local_batch_size ({local_batch_size}) must be >= 2 for GRPO (batch_size={batch_size}, num_processes={accelerator.num_processes})"

    env_make_fn = partial(make_ngoals_env, env_config, mode="test")
    env = AsyncVectorEnv([env_make_fn] * local_batch_size, shared_memory=False, context='fork')

    # reconstruct model + meta_controller and load weights

    ckpt_config = ckpt.get("config", {})
    meta_controller = MetaController(
        dim_model=dim,
        switch_temperature=ckpt_config.get("switch_temperature", 1.0),
        target_temporal_segment_len=ckpt_config.get("target_temporal_segment_len", 4),
        kl_loss_weight=ckpt_config.get("kl_loss_weight", 1.0),
        kl_loss_warmup_steps=ckpt_config.get("kl_loss_warmup_steps", 0),
    )

    transformer_class = TransformerWithResnet if modality == MODALITY_RESNET_RGB else Transformer

    transformer_kwargs = dict(
        dim=dim,
        state_embed_readout=dict(
            num_continuous=state_dim,
            readout_kwargs=dict(continuous_log_var_embed=True)
        ),
        action_embed_readout=dict(
            num_discrete=num_actions
        ),
        lower_body=dict(depth=depth, heads=heads, attn_dim_head=dim_head),
        upper_body=dict(depth=depth, heads=heads, attn_dim_head=dim_head),
        meta_controller=meta_controller,
    )
    if modality == MODALITY_RESNET_RGB:
        transformer_kwargs["use_layernorm"] = True

    model = transformer_class(**transformer_kwargs)

    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_result.missing_keys:
        accelerator.print(f"WARNING: {len(load_result.missing_keys)} missing keys (not in checkpoint): {load_result.missing_keys[:5]}...")
    if load_result.unexpected_keys:
        accelerator.print(f"WARNING: {len(load_result.unexpected_keys)} unexpected keys (in checkpoint but not model): {load_result.unexpected_keys[:5]}...")
    if not load_result.missing_keys and not load_result.unexpected_keys:
        accelerator.print(f"Loaded model from checkpoint (all keys matched)")

    # optimizer — only train the meta_controller's internal RL parameters (action proposer)

    optim = Adam(meta_controller.internal_rl_parameters(), lr=lr)

    # prepare

    model, meta_controller, optim = accelerator.prepare(model, meta_controller, optim)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_mc = accelerator.unwrap_model(meta_controller)

    unwrapped_model.eval()
    unwrapped_mc.train()

    set_requires_grad(unwrapped_model, False)
    set_requires_grad(unwrapped_mc, True)

    device = accelerator.device

    # checkpointing

    os.makedirs(output_dir, exist_ok=True)

    def store_checkpoint(step: int):
        if accelerator.is_main_process:
            ckpt_file = os.path.join(output_dir, f"grpo_checkpoint_step_{step}.pt")
            torch.save({
                "model_state_dict": unwrapped_model.state_dict(),
                "meta_controller_state_dict": unwrapped_mc.state_dict(),
                "gradient_step": step,
                "config": ckpt_config,
            }, ckpt_file)
            accelerator.print(f"Checkpoint saved to {ckpt_file}")

    # rollout loop

    num_batch_updates = num_episodes // batch_size
    group_rejections = 0

    accelerator.print(f"GRPO step offset: {step_offset} (continuing from training checkpoint)")

    pbar = tqdm(range(num_batch_updates), desc='GRPO training', disable=not accelerator.is_local_main_process)

    for gradient_step in pbar:
        global_step = step_offset + gradient_step

        env_seeds = [torch.randint(0, 1_000_000, (1,)).item() for _ in range(local_batch_size)]

        state, _ = env.reset(seed=env_seeds)

        cache = None
        past_action_id = None

        iteration_states = []
        iteration_log_probs = []
        iteration_switch_betas = []
        iteration_latent_actions = []
        iteration_rewards = []

        dones = torch.zeros(local_batch_size, dtype=torch.bool)
        all_steps_dones = []

        for step in range(max_timesteps):
            image = state['image']
            image_tensor = torch.from_numpy(image).float().to(device)

            if modality == MODALITY_RESNET_RGB:
                image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
                image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).to(device)) / torch.tensor([0.229, 0.224, 0.225]).to(device)
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')
            elif modality == MODALITY_RAW_RGB:
                image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
                image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).to(device)) / torch.tensor([0.229, 0.224, 0.225]).to(device)
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')
            elif modality == MODALITY_SYMBOLIC:
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')

            if torch.is_tensor(past_action_id):
                past_action_id = past_action_id.long()

            with torch.no_grad():
                logits, cache = unwrapped_model(
                    state=image_tensor,
                    actions=past_action_id,
                    meta_controller=unwrapped_mc,
                    return_cache=True,
                    return_raw_action_dist=True,
                    cache=cache,
                )

            logits[..., 0] = float('-inf')
            action = unwrapped_model.action_readout.sample(logits)
            past_action_id = action

            grpo_data = extract_grpo_data(unwrapped_mc, cache)

            iteration_states.append(grpo_data.state)
            iteration_log_probs.append(grpo_data.log_prob)
            iteration_switch_betas.append(grpo_data.switch_beta)
            iteration_latent_actions.append(grpo_data.action)

            env_action = (action - 1).squeeze(-1).cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(env_action)

            reward_tensor = torch.from_numpy(reward).float().to(device)
            iteration_rewards.append(rearrange(reward_tensor, 'b -> b 1'))

            all_steps_dones.append(dones.clone())
            dones |= torch.from_numpy(terminated | truncated)

            if dones.all():
                break

            state = next_state

        episode_lens = (~torch.stack(all_steps_dones)).sum(dim=0).to(device)

        cur_states = cat(iteration_states, dim=1)
        cur_log_probs = cat(iteration_log_probs, dim=1)
        cur_switch_betas = cat(iteration_switch_betas, dim=1)
        cur_latent_actions = cat(iteration_latent_actions, dim=1)
        cur_rewards = cat(iteration_rewards, dim=1)

        group_states = cur_states
        group_log_probs = cur_log_probs
        group_switch_betas = cur_switch_betas
        group_latent_actions = cur_latent_actions
        group_step_rewards = cur_rewards

        # mask rewards after done

        mask = lens_to_mask(episode_lens, group_step_rewards.shape[-1])
        group_step_rewards = group_step_rewards * mask

        cumulative_rewards = group_step_rewards.sum(dim=-1)

        # reward shaping

        shaped_rewards = reward_shaping_fn(
            cumulative_rewards,
            group_step_rewards,
            episode_lens,
            reject_threshold_cumulative_reward_variance=reject_threshold_cumulative_reward_variance
        )

        if not exists(shaped_rewards):
            accelerator.print(f'group rejected - variance {cumulative_rewards.var().item():.4f} < threshold {reject_threshold_cumulative_reward_variance}')
            group_rejections += 1
            continue

        # GRPO advantages

        all_shaped_rewards = accelerator.gather(shaped_rewards)
        all_advantages = z_score(all_shaped_rewards).float()

        process_index = accelerator.process_index
        num_local = shaped_rewards.shape[0]
        group_advantages = all_advantages[process_index * num_local: (process_index + 1) * num_local]

        if torch.any(torch.isnan(group_advantages)):
            accelerator.print(f'group rejected - advantages contained NaNs')
            group_rejections += 1
            continue

        if should_reject_group_based_on_switch_betas(group_switch_betas, episode_lens):
            accelerator.print(f'group rejected - switch betas are all zero')
            group_rejections += 1
            continue

        # PPO-style multi-epoch update

        epoch_losses = []
        epoch_policy_losses = []
        epoch_kl_losses = []
        epoch_entropy_losses = []
        epoch_grad_norms = []

        # entropy warmup: ramp from 0 to entropy_loss_weight over entropy_warmup_steps
        if entropy_warmup_steps > 0 and gradient_step < entropy_warmup_steps:
            current_entropy_weight = entropy_loss_weight * (gradient_step / entropy_warmup_steps)
        else:
            current_entropy_weight = entropy_loss_weight

        for epoch in range(num_epochs):
            optim.zero_grad()

            policy_loss, kl_loss = unwrapped_mc.policy_loss(
                group_states,
                group_log_probs,
                group_latent_actions,
                group_advantages,
                group_switch_betas,
                episode_lens=episode_lens,
                kl_loss_weight=kl_loss_weight,
                eps_clip=0.2,
                return_kl_loss=True
            )

            # entropy bonus on switch betas (encourages exploration of switching)
            switch_mask = lens_to_mask(episode_lens, group_switch_betas.shape[-1])
            entropy_loss = (binary_entropy(group_switch_betas) * switch_mask).sum() / switch_mask.sum()

            loss = policy_loss + kl_loss - current_entropy_weight * entropy_loss

            accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(meta_controller.parameters(), max_grad_norm)

            optim.step()

            epoch_losses.append(loss.item())
            epoch_policy_losses.append(policy_loss.item())
            epoch_kl_losses.append(kl_loss.item())
            epoch_entropy_losses.append(entropy_loss.item())
            epoch_grad_norms.append(grad_norm.item())

        mean_loss = np.mean(epoch_losses)
        mean_policy_loss = np.mean(epoch_policy_losses)
        mean_kl_loss = np.mean(epoch_kl_losses)
        mean_entropy_loss = np.mean(epoch_entropy_losses)
        mean_grad_norm = np.mean(epoch_grad_norms)

        pbar.set_postfix(
            loss=f'{mean_loss:.4f}',
            grad_norm=f'{mean_grad_norm:.4f}',
            reward=f'{cumulative_rewards.mean().item():.4f}',
        )

        if use_wandb and accelerator.is_main_process:
            wandb.log({
                'grpo/loss': mean_loss,
                'grpo/policy_loss': mean_policy_loss,
                'grpo/kl_loss': mean_kl_loss,
                'grpo/entropy_loss': mean_entropy_loss,
                'grpo/entropy_weight': current_entropy_weight,
                'grpo/grad_norm': mean_grad_norm,
                'grpo/reward': cumulative_rewards.mean().item(),
                'grpo/reward_std': cumulative_rewards.std().item(),
                'grpo/switch_density': group_switch_betas.mean().item(),
                'grpo/group_rejections': group_rejections,
            }, step=global_step)

        accelerator.print(f'[step {global_step}] loss: {mean_loss:.4f}, grad_norm: {mean_grad_norm:.4f}, reward: {cumulative_rewards.mean().item():.4f}')

        if gradient_step % save_steps == 0:
            store_checkpoint(gradient_step)

        if gradient_step % eval_steps == 0 and gradient_step > 0:
            visualize_switch_betas(
                switch_betas=group_switch_betas,
                episode_lens=episode_lens,
                gradient_step=global_step,
                num_samples=3,
                use_wandb=use_wandb,
                accelerator=accelerator
            )

            visualize_eval_trajectory(
                unwrapped_model=unwrapped_model,
                unwrapped_meta_controller=unwrapped_mc,
                env_config=env_config,
                modality=modality,
                gradient_step=global_step,
                max_timesteps=max_timesteps,
                seed=torch.randint(0, 1_000_000, (1,)).item(),
                use_wandb=use_wandb,
                accelerator=accelerator,
            )

    env.close()

    # save final

    store_checkpoint(num_batch_updates)

    if use_wandb and accelerator.is_main_process:
        wandb.finish()

if __name__ == '__main__':
    Fire(main)
