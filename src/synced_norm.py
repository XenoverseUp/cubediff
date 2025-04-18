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

            # Reshape to group channels and separate cubemap faces
            channels_per_group = num_channels // num_groups
            x_reshaped = x.reshape(actual_batch, 6, num_groups, channels_per_group, height, width)

            # Compute mean and variance across all faces AND spatial dimensions
            # This ensures color consistency across the entire cubemap
            mean = x_reshaped.mean(dim=(1, 4, 5), keepdim=True)
            var = x_reshaped.var(dim=(1, 4, 5), keepdim=True, unbiased=False)

            # Normalize
            x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)

            # Reshape back
            x_normalized = x_normalized.reshape(batch_size, num_channels, height, width)

            # Apply weight and bias
            return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            # Standard GroupNorm for non-cubemap inputs
            return F.group_norm(
                x,
                num_groups=min(self.num_groups, num_channels),
                weight=self.weight,
                bias=self.bias,
                eps=self.eps
            )

    def _forward_3d(self, x):
        # Handle 3D input: [batch, sequence, channels]
        batch_size, sequence_length, num_channels = x.shape

        # For 3D input from attention layers
        if batch_size % 6 == 0:
            actual_batch = batch_size // 6

            # IMPORTANT: This reshaping must match CubemapAttention
            # In CubemapAttention, x is reshaped from [B*6, L, D] to [B, 6*L, D]
            # We need to temporarily reshape differently to compute statistics
            # across faces, then reshape back to match CubemapAttention

            # First reshape to separate faces: [B*6, L, C] -> [B, 6, L, C]
            x_temp = x.reshape(actual_batch, 6, sequence_length, num_channels)

            # Compute statistics across faces only (not sequence)
            # This makes more sense for attention features
            mean = x_temp.mean(dim=1, keepdim=True)  # only across faces
            var = x_temp.var(dim=1, keepdim=True, unbiased=False)

            # Normalize while still in [B, 6, L, C] format
            x_normalized = (x_temp - mean) / torch.sqrt(var + self.eps)

            # Reshape back to original format: [B, 6, L, C] -> [B*6, L, C]
            x_normalized = x_normalized.reshape(batch_size, sequence_length, num_channels)

            # Apply weights and bias
            return x_normalized * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        else:
            # For non-cubemap inputs, use standard LayerNorm approach
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

            # Apply weights and bias
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
