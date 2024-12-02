#!/usr/bin/env python3
from transformers import T5ForConditionalGeneration

from attention import DifferentialSelfAttention, DifferentialCrossAttention

def get_model(base: str = "t5-small", patch=False):
    model = T5ForConditionalGeneration.from_pretrained(base)
    if patch:
        # patch encoder self-attention layer with diff attention
        for layer in model.encoder.block:
            layer.layer[0].SelfAttention = DifferentialSelfAttention(
                model_dim=layer.layer[0].SelfAttention.config.d_model,
                num_heads=layer.layer[0].SelfAttention.config.num_heads,
                dropout_rate=layer.layer[0].SelfAttention.dropout.p,
                has_relative_attention_bias=False,
            )

        # patch decoder self-attention + cross-attention layers with diff attention
        for layer in model.decoder.block:
            layer.layer[0].SelfAttention = DifferentialSelfAttention(
                model_dim=layer.layer[0].SelfAttention.config.d_model,
                num_heads=layer.layer[0].SelfAttention.config.num_heads,
                dropout_rate=layer.layer[0].SelfAttention.dropout.p,
                has_relative_attention_bias=False,
            )
            layer.layer[1].EncDecAttention = DifferentialCrossAttention(
                model_dim=layer.layer[1].EncDecAttention.config.d_model,
                num_heads=layer.layer[1].EncDecAttention.config.num_heads,
                dropout_rate=layer.layer[1].EncDecAttention.dropout.p,
                has_relative_attention_bias=False,
            )
    return model

def freeze_non_attention_layers(model):
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze attention layers
    for block in model.encoder.block:
        block.layer[0].SelfAttention.q.requires_grad = True
        block.layer[0].SelfAttention.k.requires_grad = True
        block.layer[0].SelfAttention.v.requires_grad = True
        block.layer[0].SelfAttention.o.requires_grad = True

    for block in model.decoder.block:
        block.layer[0].SelfAttention.q.requires_grad = True
        block.layer[0].SelfAttention.k.requires_grad = True
        block.layer[0].SelfAttention.v.requires_grad = True
        block.layer[0].SelfAttention.o.requires_grad = True

        block.layer[1].EncDecAttention.q.requires_grad = True
        block.layer[1].EncDecAttention.k.requires_grad = True
        block.layer[1].EncDecAttention.v.requires_grad = True
        block.layer[1].EncDecAttention.o.requires_grad = True
    