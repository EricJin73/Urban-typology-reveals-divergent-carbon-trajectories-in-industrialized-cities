"""
FiLM Seq2Seq Model Definition
"""
import torch
import torch.nn as nn


class FiLMSeq2Seq(nn.Module):
    """
    FiLM-based Sequence-to-Sequence model with attention mechanism.
    """
    
    def __init__(self, n_cities, emb_dim, feat_dim,
                 enc_hidden=64, dec_hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.num_layers = num_layers
        
        # City embeddings and FiLM generator
        self.city_emb = nn.Embedding(n_cities, emb_dim)
        self.film_gen = nn.Linear(emb_dim, 2 * feat_dim)
        
        # Encoder and Decoder
        self.encoder = nn.LSTM(feat_dim, enc_hidden, num_layers, 
                              batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(feat_dim, dec_hidden, num_layers, 
                              batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.use_attention = True
        self.enc_out_proj = (nn.Linear(enc_hidden, dec_hidden) 
                            if enc_hidden != dec_hidden 
                            else nn.Identity())
        self.attn_in_proj = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.head_attn = nn.Sequential(
            nn.Linear(dec_hidden * 2, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, 1)
        )
        
        # Alternate prediction head (without attention)
        self.head = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden // 2, 1)
        )
        
        # Bridge layers (if encoder and decoder dimensions differ)
        if enc_hidden != dec_hidden:
            self.bridge_h = nn.Linear(enc_hidden, dec_hidden)
            self.bridge_c = nn.Linear(enc_hidden, dec_hidden)
        else:
            self.bridge_h = self.bridge_c = None
    
    def apply_film(self, x, city_idx):
        """Apply FiLM modulation."""
        emb = self.city_emb(city_idx)  # [B, emb_dim]
        gamma, beta = self.film_gen(emb).chunk(2, dim=1)  # [B, feat_dim] each
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)
    
    def encode(self, x_enc, city_idx):
        """Encoder forward pass."""
        x_mod = self.apply_film(x_enc, city_idx)
        enc_out, (h, c) = self.encoder(x_mod)
        
        # Bridge hidden states
        if self.bridge_h is not None:
            h = self.bridge_h(h.transpose(0, 1)).transpose(0, 1)
            c = self.bridge_c(c.transpose(0, 1)).transpose(0, 1)
        
        enc_ctx = self.enc_out_proj(enc_out)
        return (h.contiguous(), c.contiguous()), enc_ctx
    
    def decode(self, x_dec, hc, city_idx, enc_ctx):
        """Decoder forward pass (with attention)."""
        x_mod = self.apply_film(x_dec, city_idx)
        dec_out, _ = self.decoder(x_mod, hc)
        
        if self.use_attention:
            # Luong attention
            q = self.attn_in_proj(dec_out)
            attn_scores = torch.bmm(q, enc_ctx.transpose(1, 2))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.bmm(attn_weights, enc_ctx)
            fused = torch.cat([dec_out, context], dim=-1)
            y = self.head_attn(fused).squeeze(-1)
        else:
            y = self.head(dec_out).squeeze(-1)
        
        return y
    
    def forward(self, x_enc, city_idx, x_dec):
        """Complete forward pass."""
        hc0, enc_ctx = self.encode(x_enc, city_idx)
        y = self.decode(x_dec, hc0, city_idx, enc_ctx)
        return y
