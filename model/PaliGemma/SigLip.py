from typing import Tuple, Optional

import torch
import torch.nn as nn

from model.lora import get_layer

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config, use_quantize:bool = False, use_lora:bool = False):
        super().__init__()
        layer = get_layer(use_quantize, use_lora, **config.lora if use_lora else {})
        self.linear = layer(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True) # hidden size -> projection dim

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class SigLipVisionModel(nn.Module):
    def __init__(self, config, use_quantize:bool=False, use_lora:bool = False):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config, use_quantize=use_quantize, use_lora=use_lora)

    def forward(self, pixel_values)->Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)

class SigLipVisionTransformer(nn.Module):
    def __init__(self, config, use_quantize:bool=False, use_lora:bool = False):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config, use_quantize=use_quantize, use_lora=use_lora)
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values) # Initial Embeddings

        last_hidden_state = self.encoder(inputs_embeds=hidden_states) # Layer L embeddings with Multi-Head attention and MLP

        last_hidden_state = self.post_layer_norm(last_hidden_state)

        return last_hidden_state

class SigLipVisionEmbeddings(nn.Module):
    """
    This class processes the image into a sequence of patches.

    Generates the Layer 0 hidden states -> input embeddings

    Config:
        hidden_size: The dimension of the embeded patch vector
        image_size: The size of the image
        patch_size: The size of the patch
        num_channels: The number of channels of the image -> 3 [RGB]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" #no padding added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(
            num_embeddings=self.num_positions,
            embedding_dim=self.embed_dim,
        )
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand(1, -1), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        (_, _, height, width) = pixel_values.shape # [Batch_Size, Num_Channels, Height, Width]

        # split the image into patches
        # flatten each patch
        # project the patches into the embed_dim
        patch_embeds = self.patch_embeddings(pixel_values) # [Batch_Size, Embed_Dim, Height // Patch_Size, Width // Patch_Size]

        # flatten the spatial dimensions
        embeddings = patch_embeds.flatten(2) # [Batch_Size, Embed_Dim, Height // Patch_Size, Width // Patch_Size] -> [Batch_Size, Embed_Dim, Num_Patches] 

        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)

        # Add positional embedding to each patch. Each pos. embedding is a vector os size [Embed_dim]
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        # [Batch_Size, Num_Patches (or) Seq Length, Embed_Dim]
        return embeddings


class SigLipEncoder(nn.Module):
    """
    Generates the Layer L hidden states -> final hidden states
    This is run to make tokens aware of the other tokens -> Context
    Input -> [Batch_Size, Num_Patches (or) Seq Length, Embed_Dim]
    Output -> [Batch_Size, Num_Patches (or) Seq Length, Embed_Dim]
    """
    def __init__(self, config, use_quantize:bool=False, use_lora:bool = False):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
                SigLipEncoderLayer(config,  use_quantize = use_quantize, use_lora = use_lora) for _ in range(config.num_hidden_layers)
            ])
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # input embeds = [Batch_Size, Num_Patches (or) Seq Length, Embed_Dim]
        hidden_states=inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

class SigLipEncoderLayer(nn.Module):
    """
    Uses Attention and MLP to generate the final hidden states
    """
    def __init__(self, config, use_quantize:bool = False, use_lora: bool = False):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config, use_quantize=use_quantize, use_lora=use_lora)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config, use_quantize=use_quantize, use_lora=use_lora)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    '''
        config:
            hidden_size
            num_dttention_heads
            attention_dropout
    '''
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        layer = get_layer(
            use_quantize,
            use_lora,
            **config.lora if use_lora else {},
        )
        self.k_proj = layer(self.embed_dim, self.embed_dim)
        self.v_proj = layer(self.embed_dim, self.embed_dim)
        self.q_proj = layer(self.embed_dim, self.embed_dim)
        self.out_proj = layer(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SigLipMLP(nn.Module):
    """Two-layer feed-forward block used inside the encoder."""
    def __init__(self, config, use_quantize:bool = False, use_lora: bool = False):
        super().__init__()
        self.config = config
        layer = get_layer(use_quantize, use_lora, **config.lora if use_lora else {})
        self.fc1 = layer(config.hidden_size, config.intermediate_size)
        self.fc2 = layer(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Project to expanded MLP dimension
        hidden_states = self.fc1(hidden_states)  # [Batch_Size, Num_Patches, Hidden] -> [Batch_Size, Num_Patches, Intermediate]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # Bring back to the model hidden size
        hidden_states = self.fc2(hidden_states)  # [Batch_Size, Num_Patches, Intermediate] -> [Batch_Size, Num_Patches, Hidden]

        return hidden_states
