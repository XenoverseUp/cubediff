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
        if batch_size % 6 == 0:
            actual_batch = batch_size // 6

            # Reshape to join faces: (B*6, L, D) -> (B, 6*L, D)
            hidden_states_reshaped = hidden_states.reshape(actual_batch, 6 * sequence_length, dim)

            # Handle encoder hidden states differently (cross attention)
            encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

            if encoder_hidden_states is not None:
                encoder_batch_size = encoder_hidden_states.shape[0]
                encoder_seq_length = encoder_hidden_states.shape[1]
                encoder_dim = encoder_hidden_states.shape[2]

                # Case 1: Per-face text embeddings (one text embedding per face)
                if encoder_batch_size == batch_size:
                    # Reshape: (B*6, S, D) -> (B, 6*S, D)
                    encoder_hidden_states = encoder_hidden_states.reshape(
                        actual_batch, 6, encoder_seq_length, encoder_dim
                    ).reshape(actual_batch, 6 * encoder_seq_length, encoder_dim)

                # Case 2: One text embedding per cubemap (same text for all faces)
                elif encoder_batch_size == actual_batch:
                    # Keep as is - each cubemap has one text embedding
                    pass

                # Case 3: Classifier-free guidance (doubled batch)
                elif encoder_batch_size == 2 * actual_batch:
                    # For CFG, we need to maintain the structure of [uncond, cond]
                    # But we need to reshape each one separately
                    uncond_embeddings = encoder_hidden_states[:actual_batch]
                    cond_embeddings = encoder_hidden_states[actual_batch:]

                    # No reshaping needed for CFG since it's already aligned with the
                    # batch dimension of the reshaped hidden states
                    # This is crucial - we maintain the original structure so the CFG calculation works

                    # The key insight: attention between the text embeddings and hidden states
                    # will work correctly because hidden_states_reshaped has batch dimension actual_batch,
                    # which matches the batch dimension of each half of encoder_hidden_states
                    pass

                # Update kwargs with processed encoder states
                kwargs["encoder_hidden_states"] = encoder_hidden_states

            # For self-attention, also reshape the attention mask if present
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None:
                # Reshape attention mask to match hidden states
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
