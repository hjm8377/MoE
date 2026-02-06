import torch
import torch.nn as nn
import torch.nn.functional as F

if float(torch.version.cuda) > 10.1:
    from tutel import moe as tutel_moe


class MoELayer(nn.Module):
    def __init__(self, in_features, hidden_features, num_experts, top_k=2):
        super().__init__()

        self.shared_moe = tutel_moe.moe_layer(
            gate_type={'type': 'cosine_top', 'k': top_k},
            model_dim = in_features,
            experts = {'type': 'ffn', 'num_experts_per_device': num_experts, 'hidden_size_per_expert': hidden_features, 'activation_fn': lambda x: F.relu(x)},
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        )

        self.day_moe = tutel_moe.moe_layer(
            gate_type={'type': 'cosine_top', 'k': top_k}, 
            model_dim = in_features,
            experts = {'type': 'ffn', 'num_experts_per_device': num_experts, 'hidden_size_per_expert': hidden_features, 'activation_fn': lambda x: F.relu(x)},
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        )

        self.night_moe = tutel_moe.moe_layer(
            gate_type={'type': 'cosine_top', 'k': top_k}, 
            model_dim = in_features,
            experts = {'type': 'ffn', 'num_experts_per_device': num_experts, 'hidden_size_per_expert': hidden_features, 'activation_fn': lambda x: F.relu(x)},
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        )

        self.register_buffer('ortho_loss', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('aux_loss', torch.tensor(0., dtype=torch.float32))

        self.domain_scale = nn.Parameter(torch.zeros(1))

    def update_loss(self, f_s, f_d):
        # compute & update orthogonality loss
        s_norm = F.normalize(f_s, p=2, dim=-1)
        d_norm = F.normalize(f_d, p=2, dim=-1)

        ortho_loss = torch.mean((s_norm.view(-1, s_norm.size(-1)) * d_norm.view(-1, d_norm.size(-1))).sum(dim=-1) ** 2)
        self.ortho_loss = ortho_loss

    def forward(self, x, domain_label, return_separate=False):
        orig_type = x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            
            x_float = x.float()
            self.shared_moe.float()
            
            f_shared = self.shared_moe(x_float)

            day_mask = (domain_label == 0).flatten()
            night_mask = (domain_label == 1).flatten()

            f_domain = torch.zeros_like(f_shared)

            aux = self.shared_moe.l_aux

            if day_mask.any():
                f_day_out = self.day_moe(x[day_mask])
                f_domain = f_domain.index_copy(0, torch.where(day_mask)[0], f_day_out)
                aux = aux + self.day_moe.l_aux

            if night_mask.any():
                f_night_out = self.night_moe(x[night_mask])
                f_domain = f_domain.index_copy(0, torch.where(night_mask)[0], f_night_out)
                aux = aux + self.night_moe.l_aux

        if self.training:
            self.aux_loss = aux
            self.update_loss(f_shared, f_domain)

        if return_separate:
            return f_shared.to(orig_type), f_domain.to(orig_type)

        out = f_shared + self.domain_scale * f_domain
        return out.to(orig_type)
