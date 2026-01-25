import os
import h5py
import torch
import argparse
from metacontroller import (
    resnet18,
    Transformer,
    xavier_init
)
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class BehaviorCloningDataset(Dataset):
    """
    Lazy-loading dataset for HDF5 trajectory files.
    Only loads individual samples on-demand to avoid OOM.
    
    Note: For multi-worker DataLoader, each worker opens its own file handle.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._h5file = None
        
        # Get dataset length without loading data into memory
        with h5py.File(file_path, "r") as f:
            self._len = f["episode_obs"].shape[0]
    
    @property
    def h5file(self):
        """Lazy open - handles multi-worker DataLoader by opening per-worker."""
        if self._h5file is None:
            self._h5file = h5py.File(self.file_path, "r")
        return self._h5file
    
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Load single sample from disk
        obs = torch.from_numpy(self.h5file["episode_obs"][idx]).permute(0, 3, 1, 2).float()  # T, C, H, W
        actions = torch.from_numpy(self.h5file["episode_act"][idx]).squeeze(-1).int()  # T
        masks = torch.from_numpy(self.h5file["episode_masks"][idx]).bool()  # T
        
        return {
            'state': obs,
            'actions': actions,
            'masks': masks
        }
    
    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()

def save_checkpoint(epoch, global_step, discovery_phase):
    """Save training checkpoint."""
    if args.checkpoint_dir is None:
        return
    checkpoint = {
        "model": model.state_dict(),
        "visual_encoder": visual_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "discovery_phase": discovery_phase,
    }
    ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, ckpt_path)
    # Also save as latest for easy resume
    latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_path", type=str,
                        help="path to trajectories tensorfile")
    parser.add_argument("--out_path", type=str, default="model_weights.pt",
                        help="path to store trained model weights")
    parser.add_argument("--cloning_epochs", type=int, default=5,
                        help="bwd passes per batch, passive cloning")
    parser.add_argument("--discovery_epochs", type=int, default=5,
                        help="bwd passes per batch, learn subgoals from cloning")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="samples per sft batch")
    parser.add_argument("--dim", type=int, default=512,
                        help="transformer dimension")
    parser.add_argument("--lower_body_depth", type=int, default=2,
                        help="depth of lower body transformer")
    parser.add_argument("--upper_body_depth", type=int, default=2,
                        help="depth of upper body transformer")
    parser.add_argument("--num_discrete_actions", type=int, default=7,
                        help="number of discrete actions")
    parser.add_argument("--use_wandb", action="store_true",
                        help="enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="behavior-cloning",
                        help="wandb project name")
    parser.add_argument("--model_lr", type=float, default=3e-4,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--resnet_lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="device to use for training")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="directory to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="save checkpoint every k steps")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="path to checkpoint to resume from")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)

    # Initialize wandb if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )

    # Data
    action_embed_readout = dict(num_discrete = args.num_discrete_actions)
    dataset = DataLoader(
        BehaviorCloningDataset(args.dataset_path), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=16
    )

    # Model
    RESNET_PATCH_DIM = 1000
    visual_encoder = resnet18().to(device)
    model = Transformer(
        dim = args.dim,
        action_embed_readout = action_embed_readout,
        state_embed_readout = dict(num_continuous = RESNET_PATCH_DIM),
        lower_body = dict(depth = args.lower_body_depth,),
        upper_body = dict(depth = args.upper_body_depth,),
    ).to(device)
    xavier_init(model)

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': visual_encoder.parameters(), 'lr': args.resnet_lr},
        {'params': model.parameters(), 'lr': args.model_lr}
    ])

    # Create checkpoint directory if specified
    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)


    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    global_step = 0
    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        start_step = global_step % len(dataset)  # step within current epoch
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    # Behavioral cloning
    discovery_phase = False
    total_epochs = args.cloning_epochs + args.discovery_epochs
    total_steps = len(dataset) * total_epochs
    progressbar = tqdm(range(total_steps), initial=global_step)
    model.train()
    visual_encoder.train()
    for epoch in range(start_epoch, total_epochs):
        if epoch >= args.cloning_epochs:
            discovery_phase = True

        for step_in_epoch, batch in enumerate(dataset):
            # Skip steps if resuming mid-epoch
            if epoch == start_epoch and step_in_epoch < start_step:
                continue
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # vis enc
            b, t, c, h, w = batch["state"].shape
            x_pre_resnet = batch["state"].reshape(b * t, c, h, w)
            x_post_resnet = visual_encoder(x_pre_resnet)
            x_post_resnet = x_post_resnet.reshape(b, t, RESNET_PATCH_DIM)
            # fwd pass
            state_clone_loss, action_clone_loss = model(state=x_post_resnet, actions=batch["actions"], mask=batch["masks"], discovery_phase=discovery_phase)
            behavior_cloning_loss = state_clone_loss + 0.5 * action_clone_loss
            optimizer.zero_grad()
            behavior_cloning_loss.backward()
            
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(visual_encoder.parameters()) + list(model.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            global_step += 1
            progressbar.update(1)
            progressbar.set_description(f"loss={behavior_cloning_loss.item():.4f}")
            
            # Log to wandb if enabled
            if args.use_wandb:
                wandb.log({
                    "loss": behavior_cloning_loss.item(),
                    "state_clone_loss": state_clone_loss.item(),
                    "action_clone_loss": action_clone_loss.item(),
                    "grad_norm": grad_norm.item(),
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            # Save checkpoint every k steps
            if args.checkpoint_dir is not None and global_step % args.checkpoint_every == 0:
                save_checkpoint(epoch, global_step, discovery_phase)

    # Save final checkpoint
    if args.checkpoint_dir is not None:
        save_checkpoint(total_epochs, global_step, discovery_phase)
        print(f"Final checkpoint saved at step {global_step}")

    # Store model
    torch.save({
        "model": model.state_dict(),
        "visual_encoder": visual_encoder.state_dict()
    }, args.out_path)
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()