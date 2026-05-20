import torch
import torch.nn as nn
from torch import tensor
from einops import einsum

from torch_einops_utils.save_load import save_load
from metacontroller.metacontroller import MetaControllerOutput


@save_load()
class EnforcedMetaController(nn.Module):
    """
    Teacher-enforced metacontroller: uses ground-truth goal boundary signals
    to deterministically select per-goal linear probes that project activations
    into a control signal added to the residual stream.

    goal_signals: (B, T) — 0 means "no goal boundary, keep activations as-is",
    positive integer k means "select U_t[k-1] and project activations".
    """

    def __init__(self, num_goals: int, embed_dim: int = None, dim: int = None, **_):
        super().__init__()
        self.num_goals = num_goals
        self.embed_dim = embed_dim if embed_dim is not None else dim
        assert self.embed_dim is not None, "must provide embed_dim or dim"

        # one linear probe per goal (indexed 1..num_goals in goal_signals)
        inner_dim = int(self.embed_dim//2)
        # self.U_t = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.embed_dim, inner_dim, bias=True),
        #         nn.Linear(inner_dim, inner_dim, bias=True),
        #         nn.Linear(inner_dim, self.embed_dim, bias=True),
        #     ) for _ in range(num_goals)
        # ])
        self.U_t = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim, bias=True) for _ in range(num_goals)
        ])

        self.register_buffer('zero', tensor(0.), persistent=False)

    def discovery_parameters(self):
        params = []
        for probe in self.U_t:
            params.extend(probe.parameters())
        return params

    def maybe_increment_kl_loss_step(self):
        pass

    def reset_kl_loss_warmup(self):
        pass

    def forward(self, activations: torch.Tensor, goal_signals: torch.Tensor = None, **kwargs):
        """
        activations: (B, T, D)
        goal_signals: (B, T) — dense integer tensor with the current subgoal label at every timestep.
                       0 = before first boundary (no control signal), k>0 selects U_t[k-1].

        Returns: (control_signal, MetaControllerOutput)
        """
        B, T, D = activations.shape
        device = activations.device

        control_signal = torch.zeros_like(activations)  # (B, T, D)

        if goal_signals is not None:
            # goal_signals=0 means "before first boundary" -> no control signal
            # goal_signals=k (k=1..num_goals) -> project through U_t[k-1]
            for k in range(1, self.num_goals + 1):
                mask_k = (goal_signals == k)  # (B, T)
                if mask_k.any():
                    projected = self.U_t[k - 1](activations)  # (B, T, D)
                    control_signal = control_signal + projected * mask_k.unsqueeze(-1).float()

        switch_beta = (goal_signals.diff(dim=1) != 0).float() if goal_signals is not None else torch.zeros(B, T, device=device)
        if goal_signals is not None:
            # prepend a 0 for the first timestep so shape stays (B, T)
            switch_beta = torch.cat([torch.zeros(B, 1, device=device), switch_beta], dim=1)

        meta_output = MetaControllerOutput(
            prev_hiddens=None,
            input_residual_stream=activations,
            action_dist=None,
            actions=None,
            switch_beta=switch_beta,
            kl_loss=self.zero,
            kl_loss_weight=self.zero,
            ratio_loss=self.zero,
        )

        return control_signal, meta_output