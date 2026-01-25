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
    def __init__(self, file_path:str):
        dataset = torch.load(file_path)
        self.observations = dataset["episode_obs"].permute(0, 1, 4, 2, 3).float() # B, T, C, H, W
        self.actions = dataset["episode_act"].squeeze(-1).int()  # B, T
        self.masks = dataset["episode_masks"].bool()
        assert self.observations.shape[0] == self.actions.shape[0]

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'state': self.observations[idx], 
            'actions': self.actions[idx],
            'masks': self.masks[idx]
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_path", type=str,
                        help="path to trajectories tensorfile")
    parser.add_argument("--out_path", type=str, default="model_weights.pt",
                        help="path to store trained model weights")
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
    args = parser.parse_args()

    # Data
    action_embed_readout = dict(num_discrete = args.num_discrete_actions)
    dataset = DataLoader(
        BehaviorCloningDataset(args.dataset_path), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=4
    )

    # Model
    RESNET_PATCH_DIM = 1000
    visual_encoder = resnet18()
    model = Transformer(
        dim = args.dim,
        action_embed_readout = action_embed_readout,
        state_embed_readout = dict(num_continuous = RESNET_PATCH_DIM),
        lower_body = dict(depth = args.lower_body_depth,),
        upper_body = dict(depth = args.upper_body_depth,),
    )
    xavier_init(model)

    # Behavioral cloning
    progressbar = tqdm(dataset)
    for batch in progressbar:
        # vis enc
        b, t, c, h, w = batch["state"].shape
        x_pre_resnet = batch["state"].reshape(b * t, c, h, w)
        x_post_resnet = visual_encoder(x_pre_resnet)
        x_post_resnet = x_post_resnet.reshape(b, t, RESNET_PATCH_DIM)
        # fwd pass
        state_clone_loss, action_clone_loss = model(state=x_post_resnet, actions=batch["actions"], mask=batch["masks"])
        behavior_cloning_loss = state_clone_loss + 0.5 * action_clone_loss
        behavior_cloning_loss.backward()
        progressbar.set_description(f"loss={behavior_cloning_loss.item():.4f}")

    # Store model
    torch.save(model.state_dict(), args.out_path)