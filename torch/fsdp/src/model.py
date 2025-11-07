"""
Model definitions for FSDP training demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def generate_causal_mask(seq_len, device):
    """
    Generate a causal mask (lower triangular matrix)
    Args:
        seq_len: sequence length
        device: device to create mask on
    Returns:
        mask: (1, 1, seq_len, seq_len) - 1 for allowed positions, 0 for masked positions
    """
    # Create lower triangular matrix (including diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        Returns:
            normalized_x: same shape as x
        """
        # Calculate RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        normalized = x / rms * self.weight
        return normalized


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embedding to input tensor
    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos: (1, 1, seq_len, head_dim // 2) or (seq_len, head_dim // 2)
        sin: (1, 1, seq_len, head_dim // 2) or (seq_len, head_dim // 2)
    Returns:
        rotated_x: same shape as x
    """
    # Split x into pairs of dimensions
    x1, x2 = x[..., 0::2], x[..., 1::2]  # (batch, num_heads, seq_len, head_dim // 2)
    
    # Ensure cos and sin have the right shape
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    
    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Interleave back
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)  # (batch, num_heads, seq_len, head_dim // 2, 2)
    rotated_x = rotated_x.flatten(-2)  # (batch, num_heads, seq_len, head_dim)
    
    return rotated_x


class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies: theta_i = base^(-2i/d) for i in [0, d//2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)
        
    def _build_cache(self, max_seq_len):
        """Build cache for cos and sin values"""
        seq_len = max_seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # Compute outer product: t[:, None] * inv_freq[None, :]
        freqs = t[:, None] * self.inv_freq[None, :]  # (seq_len, head_dim // 2)
        
        # Cache cos and sin
        cos = freqs.cos()  # (seq_len, head_dim // 2)
        sin = freqs.sin()  # (seq_len, head_dim // 2)
        
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
        
    def forward(self, seq_len):
        """
        Get cos and sin for given sequence length
        Args:
            seq_len: sequence length
        Returns:
            cos: (seq_len, head_dim // 2)
            sin: (seq_len, head_dim // 2)
        """
        if seq_len > self.max_seq_len:
            # Rebuild cache if needed
            self._build_cache(seq_len)
        
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with causal mask and RoPE"""
    def __init__(self, hidden_size, num_heads, max_seq_len=2048):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
        # RoPE
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)
        
    def forward(self, query, key, value):
        """
        Args:
            query: (batch_size, seq_len, hidden_size)
            key: (batch_size, seq_len, hidden_size)
            value: (batch_size, seq_len, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Project Q, K, V and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch, num_heads, seq_len, head_dim)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # Apply RoPE to Q and K
        cos, sin = self.rope(seq_len)  # (seq_len, head_dim // 2)
        Q = apply_rotary_pos_emb(Q, cos, sin)
        K = apply_rotary_pos_emb(K, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, num_heads, seq_len, seq_len)
        
        # Apply causal mask
        causal_mask = generate_causal_mask(seq_len, scores.device)  # (1, 1, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )  # (batch, seq_len, hidden_size)
        
        # Output projection
        output = self.W_o(attn_output)  # (batch, seq_len, hidden_size)
        
        return output, attention_weights


class SwiGLU(nn.Module):
    """SwiGLU activation function: Swish(xW + b) ⊙ (xV + c)"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        # Swish activation: x * sigmoid(x)
        gate = F.silu(self.gate_proj(x))  # silu is Swish
        up = self.up_proj(x)
        # Element-wise multiplication
        output = gate * up
        output = self.down_proj(output)
        return output


class TransformerBlock(nn.Module):
    """Transformer block for auto-wrapping demonstration"""
    def __init__(self, hidden_size=512, num_heads=8, intermediate_size=2048, max_seq_len=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads, max_seq_len=max_seq_len)
        self.swiglu = SwiGLU(hidden_size, intermediate_size)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        # Feed-forward with SwiGLU
        ff_out = self.swiglu(x)
        x = x + ff_out
        x = self.norm2(x)
        return x


class DemoModel(nn.Module):
    """Demo model for FSDP training"""
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=512,
        num_heads=8,
        num_layers=1, 
        intermediate_size=2048,
        max_seq_len=128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # No need for positional encoding since we use RoPE
        
        # Transformer blocks (will be auto-wrapped by FSDP)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) 
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Apply RMSNorm before lm_head
        x = self.norm(x)
        x = self.lm_head(x)
        # Return all positions: (batch_size, seq_len, vocab_size)
        return x
