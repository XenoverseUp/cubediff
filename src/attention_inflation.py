import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

class CubemapAttention(nn.Module):
    """
    A wrapper for standard attention that handles cubemap faces together.
    This gets applied to the UNet's attention layers.
    """
    def __init__(self, original_attention):
        super().__init__()
        self.original_attention = original_attention

    def forward(self, hidden_states, **kwargs):
        batch_size, sequence_length, dim = hidden_states.shape

        # Check if this input contains cubemap faces (batch size divisible by 6)
        # Also handle classifier-free guidance which doubles the batch size
        if batch_size % 6 == 0:
            # Process cubemap together for consistency across faces
            actual_batch = batch_size // 6

            # Reshape to join faces: (B*6, L, D) -> (B, 6*L, D)
            hidden_states_reshaped = hidden_states.reshape(actual_batch, 6 * sequence_length, dim)

            # Handle encoder hidden states differently (cross attention)
            encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

            if encoder_hidden_states is not None:
                # For cross-attention, keep text embeddings as is
                # The original attention will handle it correctly
                result = self.original_attention(hidden_states_reshaped, **kwargs)
            else:
                # For self-attention, also reshape the attention mask if present
                attention_mask = kwargs.get("attention_mask", None)
                if attention_mask is not None:
                    # Reshape attention mask to match reshaped hidden states
                    attention_mask = attention_mask.reshape(actual_batch, -1)
                    kwargs["attention_mask"] = attention_mask

                # Call original attention with reshaped states
                result = self.original_attention(hidden_states_reshaped, **kwargs)

            # Reshape result back to original format: (B, 6*L, D) -> (B*6, L, D)
            result = result.reshape(batch_size, sequence_length, dim)

            return result
        else:
            # For non-cubemap inputs, use normal attention
            return self.original_attention(hidden_states, **kwargs)


def inflate_attention_layers(unet_model):
    """
    Replace attention modules in UNet with cubemap-aware versions
    """
    # Find and replace all self-attention modules
    for name, module in unet_model.named_modules():
        # Target all attention blocks that perform self or cross attention
        if isinstance(module, Attention):
            # Create a cubemap attention wrapper
            cubemap_attn = CubemapAttention(module)

            # Replace the module with our wrapper
            # We need to find the parent module and replace its attribute
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name

            if parent_name:
                parent = unet_model.get_submodule(parent_name)
                setattr(parent, child_name, cubemap_attn)
            else:
                # This is a top-level attribute
                setattr(unet_model, child_name, cubemap_attn)

    return unet_model
