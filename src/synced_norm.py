import torch
import torch.nn as nn



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
        # Check input dimensionality
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

        # Check if divisible by 6 (cubemap faces)
        if batch_size % 6 == 0:
            actual_batch = batch_size // 6

            # Reshape to separate faces
            x_reshaped = x.view(actual_batch, 6, num_channels, height, width)

            # Compute group normalization statistics across faces
            num_groups = min(self.num_groups, num_channels)
            group_size = num_channels // num_groups

            # Reshape for group normalization
            x_grouped = x_reshaped.view(actual_batch, 6, num_groups, group_size, height, width)

            # Compute mean and variance across spatial dimensions and faces
            mean = x_grouped.mean(dim=[1, 4, 5], keepdim=True)
            var = x_grouped.var(dim=[1, 4, 5], keepdim=True, unbiased=False)

            # Normalize
            x_normalized = (x_grouped - mean) / torch.sqrt(var + self.eps)

            # Reshape back
            x_normalized = x_normalized.view(batch_size, num_channels, height, width)
        else:
            # Standard GroupNorm for non-cubemap inputs
            x_reshaped = x.view(batch_size, num_groups, -1)
            mean = x_reshaped.mean(dim=2, keepdim=True)
            var = x_reshaped.var(dim=2, keepdim=True, unbiased=False)
            x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)
            x_normalized = x_normalized.view(batch_size, num_channels, height, width)

        # Apply scale and bias
        return x_normalized * self.weight.view(1, num_channels, 1, 1) + self.bias.view(1, num_channels, 1, 1)

    def _forward_3d(self, x):
        # Handle 3D input: [batch, sequence, channels]
        batch_size, sequence_length, num_channels = x.shape

        # Check if divisible by 6 (cubemap faces)
        if batch_size % 6 == 0:
            actual_batch = batch_size // 6

            # Reshape to separate faces
            x_reshaped = x.view(actual_batch, 6, sequence_length, num_channels)

            # Compute normalization statistics across faces
            mean = x_reshaped.mean(dim=[1, 2], keepdim=True)
            var = x_reshaped.var(dim=[1, 2], keepdim=True, unbiased=False)

            # Normalize
            x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)

            # Reshape back
            x_normalized = x_normalized.view(batch_size, sequence_length, num_channels)
        else:
            # Standard LayerNorm for non-cubemap inputs
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply scale and bias (reshape to match 3D input)
        return x_normalized * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)


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
