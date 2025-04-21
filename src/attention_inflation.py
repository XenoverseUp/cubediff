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

        if batch_size % 6 == 0:
            actual_batch = batch_size // 6

            # (B*6, L, D) -> (B, 6*L, D)
            hidden_states = hidden_states.reshape(actual_batch, 6 * sequence_length, dim)

            encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
            if encoder_hidden_states is not None and encoder_hidden_states.shape[0] == batch_size:
                encoder_hidden_states = encoder_hidden_states.reshape(
                    actual_batch, 6 * encoder_hidden_states.shape[1], encoder_hidden_states.shape[2]
                )
                kwargs["encoder_hidden_states"] = encoder_hidden_states

            result = self.original_attention(hidden_states, **kwargs)
            result = result.reshape(batch_size, sequence_length, dim)
            return result
        else:
            return self.original_attention(hidden_states, **kwargs)

def inflate_attention_layers(unet_model):
    """
    Replace attention modules in UNet with cubemap-aware versions
    """
    for name, module in unet_model.named_modules():
        if isinstance(module, Attention):
            cubemap_attn = CubemapAttention(module)

            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name

            if parent_name:
                parent = unet_model.get_submodule(parent_name)
                setattr(parent, child_name, cubemap_attn)
            else:
                setattr(unet_model, child_name, cubemap_attn)

    return unet_model
