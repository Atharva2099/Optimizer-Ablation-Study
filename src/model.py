"""
GPT-2 Style Transformer Model for Optimizer Comparison
------------------------------------------------------

A clean, configurable decoder-only transformer implementation based on the GPT-2 
architecture. Built from scratch for small-scale pretraining experiments.

Mathematical Foundation:
========================

The model implements an autoregressive language model using the transformer 
decoder stack. Given a sequence of tokens x = (x_1, x_2, ..., x_T), the model 
learns to predict the next token: P(x_t | x_1, ..., x_{t-1}).

Architecture Overview:
----------------------
1. Token embeddings: convert token IDs to dense vectors
2. Positional embeddings: add position information  
3. N transformer blocks (each containing self-attention + MLP)
4. Final layer normalization
5. Language modeling head (projection to vocab size)

Key Math: Self-Attention
------------------------
For each head, given input X ∈ R^(T×d):

    Q = X W_q,  K = X W_k,  V = X W_v     (linear projections)
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
where:
- d_k = d / num_heads (dimension per head)
- The √d_k scaling prevents dot products from growing too large,
  which would push softmax into extremely sharp regions (low gradient)

Causal Masking:
---------------
For autoregressive generation, we prevent attending to future tokens:
    mask[i,j] = -∞ if j > i, else 0
    
This makes the attention matrix lower-triangular, ensuring position i only 
attends to positions 1 through i.

Key Math: MLP Block
-------------------
The feed-forward network expands then contracts the representation:

    FFN(x) = W_down · GELU(W_up · x + b_up) + b_down
    
where:
- W_up: d_model × (4×d_model)  [expansion]
- W_down: (4×d_model) × d_model  [contraction]
- GELU: Gaussian Error Linear Unit, smoother than ReLU

The 4× expansion is the standard GPT-2 ratio. This is where Aurora's 
"tall matrix" problem appears — W_up is a tall matrix when d_model is small.

Key Math: Layer Normalization
-----------------------------
    LN(x) = γ · (x - μ) / √(σ² + ε) + β
    
where μ and σ² are computed over the feature dimension (not batch or sequence).
This stabilizes training by keeping activations in a consistent range.

Key Math: GELU Activation
-------------------------
    GELU(x) = x · Φ(x) = x · ½[1 + erf(x/√2)]
    
Approximation used in practice:
    GELU(x) ≈ 0.5 · x · (1 + tanh[√(2/π) · (x + 0.044715 · x³)])

Why GELU over ReLU? Smoother gradients, non-zero almost everywhere, 
performs better in deep transformers empirically.

Model Configurations:
---------------------
We support multiple model sizes via config dictionaries:
- 10M params: 6 layers, 6 heads, 384 dim (debugging)
- 60M params: 8 layers, 12 heads, 768 dim (main experiment)
- 125M params: 12 layers, 12 heads, 768 dim (stretch goal)

Parameter counting formula (ignoring biases, embeddings, head):
    params ≈ n_layers × (4 × d_model² + 2 × 4 × d_model²)
           = n_layers × 12 × d_model²  (simplified)
    
The 12× comes from:
- Attention: 4 × d_model² (Q, K, V, O projections)
- MLP: 8 × d_model² (up: 4×, down: 4×)

References:
-----------
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)
- NanoGPT by Andrej Karpathy (clean educational implementation)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT model sizes.
    
    Attributes:
        block_size: Maximum sequence length (context window)
        vocab_size: Size of the token vocabulary (50257 for GPT-2)
        n_layer: Number of transformer blocks
        n_head: Number of attention heads per layer
        n_embd: Model dimension (embedding size)
        dropout: Dropout rate for regularization
        bias: Whether to use bias in Linear layers and LayerNorm
    """
    block_size: int = 256      # Sequence length for our experiments
    vocab_size: int = 50257    # GPT-2 vocabulary size
    n_layer: int = 8           # Number of transformer layers
    n_head: int = 12           # Number of attention heads
    n_embd: int = 768          # Embedding dimension (d_model)
    dropout: float = 0.0       # Dropout (0 for pretraining experiments)
    bias: bool = True          # Use bias terms


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional bias.
    
    Math:
    -----
    For input X ∈ R^(B×T×d), we compute:
    
    1. Project to Q, K, V:
       Q = X W_q, K = X W_k, V = X W_v
       where W_q, W_k, W_v ∈ R^(d×d)
       
    2. Split into heads:
       Q_h ∈ R^(B×H×T×d_h) where d_h = d/H
       
    3. Compute attention per head:
       A = softmax(Q_h K_h^T / √d_h + mask) V_h
       
    4. Concatenate heads and project:
       out = concat(A_1, ..., A_H) W_o
       
    The causal mask ensures autoregressive property.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # d_h = d/H
        
        # Key, Query, Value projections (combined for efficiency)
        # Input: (B, T, d) → Output: (B, T, 3d)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        # Input: (B, T, d) → Output: (B, T, d)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: precomputed as buffer so it's not a parameter
        # Shape: (1, 1, T, T) - broadcastable to (B, H, T, T)
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # Batch, Time (seq_len), Channels (n_embd)
        
        # Compute Q, K, V projections
        # qkv shape: (B, T, 3*C) → split into 3 tensors of (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        # (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention: (B, H, T, d_h) @ (B, H, d_h, T) → (B, H, T, T)
        # Scale by 1/√d_h to prevent dot products from exploding
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: set future positions to -inf before softmax
        # This ensures position i can only attend to positions ≤ i
        att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax normalizes attention weights to sum to 1
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values: (B, H, T, T) @ (B, H, T, d_h) → (B, H, T, d_h)
        y = att @ v
        
        # Concatenate heads: (B, H, T, d_h) → (B, T, H, d_h) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation.
    
    Math:
    -----
    Standard transformer MLP with expansion ratio 4:
    
        h = GELU(x W_up + b_up)
        out = h W_down + b_down
        
    Where:
        W_up: (d_model, 4×d_model)   [tall matrix when d_model small]
        W_down: (4×d_model, d_model) [wide matrix]
    
    The tall matrix W_up is where Aurora's mechanism should matter most.
    With d_model=768, W_up is 768×3072 (tall, more rows than cols relative
    to typical square matrices).
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion: d_model → 4*d_model
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # Contraction: 4*d_model → d_model
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Expand: (B, T, d) → (B, T, 4d)
        x = self.c_fc(x)
        
        # Apply GELU activation
        x = F.gelu(x)
        
        # Contract: (B, T, 4d) → (B, T, d)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
    
    Math:
    -----
    The pre-normalization architecture (used in GPT-2 and modern variants):
    
        x' = x + Attention(LayerNorm(x))
        out = x' + MLP(LayerNorm(x'))
        
    Pre-normalization is more stable than post-normalization for deep networks.
    Each block preserves the input shape: (B, T, d) → (B, T, d)
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Self-attention with pre-normalization and residual
        x = x + self.attn(self.ln_1(x))
        
        # MLP with pre-normalization and residual
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Complete GPT model for language modeling.
    
    Architecture:
    -------------
    1. Token embeddings: vocab_size × d_model lookup table
    2. Positional embeddings: block_size × d_model lookup table
    3. N × TransformerBlock (attention + MLP)
    4. Final layer normalization
    5. LM head: d_model × vocab_size (shares weights with token embeddings)
    
    Weight Tying:
    -------------
    The input token embeddings (wte) and output LM head share weights.
    This reduces parameters by ~vocab_size × d_model and often improves
    performance by ensuring input/output spaces are aligned.
    
    Loss:
    -----
    Cross-entropy loss on next-token prediction:
        L = -Σ log P(x_t | x_{<t})
        
    We predict logits for position t using information from positions < t only
    (enforced by causal masking).
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding table (weight tied with lm_head)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embedding table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: lm_head shares weights with token embeddings
        # This is standard in GPT-2 and reduces parameters significantly
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections (GPT-2 style)
        # This accounts for the residual connection accumulation
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # Scale by 1/√(2×n_layer) to prevent gradient explosion
                # in deep residual networks
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        """Initialize weights with small normal distribution (GPT-2 style).
        
        Standard deviation 0.02 is approximately 1/√d_model for d_model=768.
        This keeps initial activations in a reasonable range.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Args:
            idx: Token indices of shape (batch_size, seq_len)
            targets: Target token indices of shape (batch_size, seq_len)
                    If provided, compute loss. If None, return logits only.
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar cross-entropy loss (if targets provided, else None)
        """
        device = idx.device
        B, T = idx.size()
        
        # Check sequence length doesn't exceed context window
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Token embeddings: (B, T) → (B, T, d)
        tok_emb = self.wte(idx)
        
        # Positional embeddings: (T,) → (T, d) → broadcast to (B, T, d)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        
        # Combine token and position embeddings
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary: (B, T, d) → (B, T, vocab_size)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens if present
            )
        
        return logits, loss
    
    def count_parameters(self) -> int:
        """Count total trainable parameters.
        
        Note: Due to weight tying between wte and lm_head, we don't double-count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Generate tokens autoregressively.
        
        Args:
            idx: Starting token sequence (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = standard, <1 = greedy, >1 = random)
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            Extended sequence (B, T+max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx if idx.size(1) <= self.config.block_size else \
                      idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus on last token prediction
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def get_model_config(size: str) -> GPTConfig:
    """Get predefined model configuration by size name.
    
    Args:
        size: One of 'tiny', 'small', 'medium'
        
    Returns:
        GPTConfig instance
        
    Configurations:
    - tiny: ~10M params, 6 layers, 6 heads, 384 dim
    - small: ~60M params, 8 layers, 12 heads, 768 dim  
    - medium: ~125M params, 12 layers, 12 heads, 768 dim
    """
    configs = {
        'tiny': GPTConfig(
            block_size=256,
            vocab_size=50257,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True
        ),
        'small': GPTConfig(
            block_size=256,
            vocab_size=50257,
            n_layer=8,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True
        ),
        'medium': GPTConfig(
            block_size=256,
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True
        )
    }
    
    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")
    
    return configs[size]


# Example usage / quick test
if __name__ == "__main__":
    # Create tiny model for quick parameter count verification
    config = get_model_config('tiny')
    model = GPT(config)
    
    print(f"Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Context window: {config.block_size} tokens")
    
    # Verify forward pass works
    batch_size, seq_len = 2, 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(x, x)  # Self-supervised: predict next token
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Verify generation works
    start_tokens = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20)
    print(f"  Generated shape: {generated.shape}")
    print("\nAll tests passed!")
