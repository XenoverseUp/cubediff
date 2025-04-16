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

        # Check if batch size is divisible by 6 (cubemap faces)
        if batch_size % 6 == 0:
            actual_batch = batch_size // 6
            num_groups = min(self.num_groups, num_channels)

            # Reshape to group channels
            channels_per_group = num_channels // num_groups
            x_reshaped = x.reshape(actual_batch, 6, num_groups, channels_per_group, height, width)

            # Compute mean and variance across all spatial dimensions and faces
            # but separately for each group
            mean = x_reshaped.mean(dim=(4, 5), keepdim=True)  # Spatial dims
            mean = mean.mean(dim=1, keepdim=True)  # Faces dim

            var = ((x_reshaped - mean) ** 2).mean(dim=(4, 5), keepdim=True)  # Spatial dims
            var = var.mean(dim=1, keepdim=True)  # Faces dim

            # Normalize
            x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)

            # Reshape back
            x_normalized = x_normalized.reshape(batch_size, num_channels, height, width)

            # Apply weight and bias
            return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            # Standard GroupNorm for non-cubemap inputs
            num_groups = min(self.num_groups, num_channels)
            return F.group_norm(
                x,
                num_groups=num_groups,
                weight=self.weight,
                bias=self.bias,
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
