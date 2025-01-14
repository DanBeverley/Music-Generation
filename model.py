from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeAttention(nn.Module):
    """
        Relative self-attention mechanism.

        Args:
            d_model: The input/output dimension of the model.
            num_heads: The number of attention heads.
        """
    def __init__(self, d_model:int,
                       num_heads:int,
                       max_seq_len:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim  = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Er = nn.Parameter(torch.randn(max_seq_len*2-1, self.head_dim)) # Relative position embedding
        self.Wo = nn.Linear(d_model, d_model) # Turn the context vector to desired output dimension

        self.logger = logging.getLogger(__name__)
    def forward(self, q:torch.Tensor,
                      k:torch.Tensor,
                      v:torch.Tensor,
                      mask:Optional[torch.Tensor]=None)->torch.Tensor:
        """
        Forward pass of the relative attention mechanism.

        Args:
            q: Query tensor of shape (B, T, d_model).
            k: Key tensor of shape (B, T, d_model).
            v: Value tensor of shape (B, T, d_model).
            mask: Attention mask of shape (1, 1, T, T) or None.
            B: Batch size
            T: Sequence Length
            H: Number of heads
            _: Placeholder

        Returns:
            Context vector of shape (B, T, d_model).
        """
        B, T, _ = q.shape
        H = self.num_heads

        q = self.Wq(q).view(B, T, H, self.head_dim).transpose(1,2)  # (B,H,T, d_k)
        k = self.Wk(k).view(B, T, H, self.head_dim).transpose(1,2)
        v = self.Wv(v).view(B, T, H, self.head_dim).transpose(1,2)

        # Relative position attention
        QEr = torch.matmul(q, self.Er.transpose(0,1)) # (B, H, T, 2T-1)
        scores = torch.matmul(q,k.transpose(2,3))                 # (B, H, T, 2T-1)
        scores_relative = self._relative_shift(QEr)
        scores = scores+scores_relative

        if mask is not None:
            scores = scores.masked_fill(mask[:,:,:T,:T]==0, float('-inf'))
        attention = F.softmax(scores/(self.head_dim**.5), dim=-1)
        context = torch.matmul(attention, v).transpose(1,2).contiguous().view(B,T,self.d_model) # (B,T,d_model)
        self.logger.debug(f"Attention weights: {attention}")
        return self.Wo(context)

    def _relative_shift(self, x:torch.Tensor)->torch.Tensor:
        """
        Performs relative shifting for relative attention.

        Args:
            x: Input tensor.
        Returns:
            Shifted tensor.
        """
        batch_size, num_heads, seq_length, _ = x.shape
        x_padded = F.pad(x, (0,0,0,1))
        x_reshaped = x_padded.view(batch_size, num_heads, seq_length+1, seq_length)
        return x_reshaped[:,:,1:,:]


class MusicTransformerEncoderLayer(nn.Module):
   """
   Music Transformer encoder layer.

   Args:
       d_model: The input/output dimension of the model.
       num_heads: The number of attention heads.
       dff: The dimension of the feed-forward network.
       dropout_rate: The dropout rate.
   """
   def __init__(self, d_model:int,
                num_heads:int,
                dff:int,
                dropout_rate:float,
                max_seq_len:int):
       super().__init__()
       self.self_attn = RelativeAttention(d_model, num_heads, max_seq_len)
       self.ffn = nn.Sequential(nn.Linear(d_model, dff),
                                nn.ReLU(),
                                nn.Linear(dff, d_model)) # Feed-forward upwards projection file
       self.norm1 = nn.LayerNorm(d_model)
       self.norm2 = nn.LayerNorm(d_model)
       self.dropout = nn.Dropout(dropout_rate)
   def forward(self, x:torch.Tensor,
               mask:torch.Tensor)->torch.Tensor:
       """
       Forward pass of the encoder layer.

       Args:
           x: Input tensor of shape (B, T, d_model).
           mask: Attention mask of shape (1, 1, T, T).

       Returns:
           Output tensor of shape (B, T, d_model).
       """
       attn_output = self.self_attn(x, x, x, mask)
       attn_output = self.dropout(attn_output)
       x = self.norm1(x+attn_output)
       ffn_output = self.ffn(x)
       ffn_output = self.dropout(ffn_output)
       x = self.norm2(x + ffn_output)
       return x

class MusicTransformer(nn.Module):
    """
    Music Transformer model.

    Args:
        num_classes: The number of classes (vocabulary size).
        d_model: The input/output dimension of the model.
        num_layers: The number of encoder layers.
        num_heads: The number of attention heads.
        dff: The dimension of the feed-forward network.
        dropout_rate: The dropout rate.
        max_seq_len: The maximum sequence length.
    """
    def __init__(self, num_classes:int,
                 d_model:int,
                 num_layers:int,
                 num_heads:int,
                 dff:int,
                 dropout_rate:float,
                 max_seq_len:int,
                 pad_token:int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([MusicTransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
                                     for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token

        self._init_weights() # Weight Initialization

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Music Transformer.

        Args:
            x: Input tensor of shape (B, T).

        Returns:
            Output tensor of shape (B, T, num_classes).
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B,T)
        x = self.embedding(x) + self.pos_embedding(pos)

        casual_mask = torch.tril(torch.ones((T, T), device = x.device))
        padding_mask = (x[:,:,0] != self.pad_token).unsqueeze(1).unsqueeze(2)
        mask = casual_mask.unsqueeze(0).unsqueeze(0)*padding_mask

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            self.logger.debug(f"Output after layer {i+2}: {x}")
        return self.fc(x)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


