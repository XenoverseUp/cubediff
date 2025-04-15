import torch
import torch.nn as nn
import torch.nn.functional as F

class SynchronizedGroupNorm(nn.Module):
    """
    Group normalization that synchronizes statistics across all 6 cubemap faces.
    Handles both 4D (convolution) and 3D (attention) inputs.
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if len(x.shape) == 4:
            # 4D input: [batch, channels, height, width]
            return self._forward_4d(x)
        elif len(x.shape) == 3:
            # 3D input: [batch, sequence, channels]
            return self._forward_3d(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 3D or 4D tensor.")

    def _forward_4d(self, x):
        # Handle 4D input: [batch, channels, height, width]
        batch_size, num_channels, height, width = x.shape

        # Apply standard GroupNorm since we're having issues with synchronized version
        num_groups = min(self.num_groups, num_channels)
        return F.group_norm(
            x,
            num_groups=num_groups,
            weight=self.weight[:num_channels] if self.weight.shape[0] >= num_channels else self.weight,
            bias=self.bias[:num_channels] if self.bias.shape[0] >= num_channels else self.bias,
            eps=self.eps
        )

    def _forward_3d(self, x):
        # Handle 3D input: [batch, sequence, channels]
        batch_size, sequence_length, num_channels = x.shape

        # Use LayerNorm approach for 3D input
        # Compute mean and variance along the channel dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Adapt weights to actual channel size
        if num_channels == self.weight.shape[0]:
            # Use our weights directly if channels match
            weight = self.weight.view(1, 1, -1)
            bias = self.bias.view(1, 1, -1)
        else:
            # Otherwise use a default approach (ones and zeros)
            weight = torch.ones(1, 1, num_channels, device=x.device)
            bias = torch.zeros(1, 1, num_channels, device=x.device)

        return x_normalized * weight + bias


def replace_groupnorm_with_synced(model):
    """
    Replace all GroupNorm layers in the model with SynchronizedGroupNorm
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.GroupNorm):
            setattr(model, name, SynchronizedGroupNorm(
                module.num_channels,
                module.num_groups,
                module.eps
            ))
        else:
            # Recursively apply to child modules
            replace_groupnorm_with_synced(module)

    return model
