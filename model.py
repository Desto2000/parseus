# model.py
""" Model definition for Transformer VAE with Static Prediction Head """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """ Standard Transformer Positional Encoding """
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: Shape [seq_len, batch_size, embedding_dim] """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SwigGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) module.

    This implements the SwiGLU activation as described in papers like "GLU Variants Improve Transformer"
    but using Swish (SiLU) activation instead of Sigmoid for the gating mechanism.

    The computation performed is:
        SwiGLU(x, W, V, b, c) = SiLU(xW + b) ⊗ (xV + c)
    where ⊗ is element-wise multiplication.
    """

    def __init__(self, dim):
        """
        Initialize the SwiGLU module.

        Args:
            dim (int): Input and Output dimension.
        """
        super(SwigGLU, self).__init__()

        # Create two linear projections - one for the gate and one for the value
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the SwiGLU module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, ..., input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, ..., output_dim]
        """
        # Gate branch - apply SiLU activation (Swish)
        gate = F.silu(self.gate_proj(x))

        # Value branch - linear projection
        value = self.value_proj(x)

        # Element-wise multiplication
        return gate * value

class TransformerVAEStaticHead(nn.Module):
    def __init__(self, num_questions, num_categories, embed_dim, latent_dim,
                 hidden_dim_dec, hidden_dim_static_head,
                 transformer_nhead, transformer_nlayers, transformer_dim_feedforward):
        super().__init__()
        self.num_questions = num_questions
        self.num_categories = num_categories
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Embeddings (Concatenation)
        self.answer_embedding = nn.Embedding(num_questions * 2, embed_dim)
        self.question_signature_embedding = nn.Embedding(num_questions, embed_dim)
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        combined_embed_factor = 3
        combined_embed_dim = embed_dim * combined_embed_factor

        self.category_relations = nn.Parameter(torch.eye(num_categories), requires_grad=True)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(combined_embed_dim, max_len=num_questions)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=combined_embed_dim, nhead=transformer_nhead,
                                                   dim_feedforward=transformer_dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)

        # VAE Head (from pooled Transformer output)
        self.fc_mu = nn.Linear(combined_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_embed_dim, latent_dim)

        # --- Attention Decoder Components ---
        # Project latent vector 'z' to the dimension of encoder outputs for attention query
        self.fc_z_proj = nn.Linear(latent_dim, combined_embed_dim)
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=combined_embed_dim,
                                               num_heads=transformer_nhead, # Use same number of heads as encoder
                                               batch_first=True)
        # --- End Attention Decoder Components ---

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim + combined_embed_dim, hidden_dim_dec // 2)
        self.decoder_fc2 = nn.Linear(hidden_dim_dec // 2, hidden_dim_dec)
        self.decoder_fc_out = nn.Linear(hidden_dim_dec, num_questions)

        # Static Prediction Head (takes mu as input)
        self.static_pred_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_static_head),
            SwigGLU(hidden_dim_static_head),
            nn.Linear(hidden_dim_static_head, num_categories),
            nn.Softmax(dim=1)  # Softmax for static distribution prediction
        )

    def _get_combined_embeddings(self, answers, categories_tensor):
        batch_size = answers.shape[0]; device = answers.device
        # Ensure categories_tensor is on the correct device
        categories = categories_tensor.to(device).unsqueeze(0).expand(batch_size, -1)
        question_indices = torch.arange(self.num_questions, device=device).unsqueeze(0).expand(batch_size, -1)
        answer_ids = question_indices * 2 + answers
        ans_embed = self.answer_embedding(answer_ids)
        q_sig_embed = self.question_signature_embedding(question_indices)
        cat_embed = self.category_embedding(categories)
        combined_embed = torch.cat([ans_embed, q_sig_embed, cat_embed], dim=-1)
        return combined_embed

    def encode(self, x_combined_embed):
        """ Encodes using Transformer. Output shape [B, EmbDim] -> mu, logvar """
        src = x_combined_embed.permute(1, 0, 2) # [SeqLen, B, EmbDim] for pos_encoder
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # [B, SeqLen, EmbDim] back for Transformer layer
        transformer_output = self.transformer_encoder(src) # Output [B, SeqLen, EmbDim]
        pooled_output = transformer_output.mean(dim=1) # [B, EmbDim]

        mu = self.fc_mu(pooled_output)
        logvar = self.fc_logvar(pooled_output)
        return mu, logvar, transformer_output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        recon_logits = self.decoder_fc_out(h2)
        return recon_logits

    def forward(self, answers, categories_tensor):
        combined_embed = self._get_combined_embeddings(answers, categories_tensor)
        mu, logvar, encoder_outputs = self.encode(combined_embed)
        z = self.reparameterize(mu, logvar)
        # 4. Prepare for Attention Decoder
        # Project z to match encoder output dimension for the query
        query = self.fc_z_proj(z) # Shape: [B, CombinedEmbedDim]
        # Add sequence dimension for MultiheadAttention: [B, 1, CombinedEmbedDim]
        query = query.unsqueeze(1)

        # 5. Apply Attention
        # Query: Projected z [B, 1, CombinedEmbedDim]
        # Key: Encoder outputs [B, NumQuestions, CombinedEmbedDim]
        # Value: Encoder outputs [B, NumQuestions, CombinedEmbedDim]
        context_vector, attn_weights = self.attention(query=query,
                                                      key=encoder_outputs,
                                                      value=encoder_outputs)
        # Output context_vector shape: [B, 1, CombinedEmbedDim]
        # Remove the sequence dimension: [B, CombinedEmbedDim]
        context_vector = context_vector.squeeze(1)

        # 6. Combine z and context for Decoder input
        decoder_input = torch.cat((z, context_vector), dim=1) # Shape: [B, LatentDim + CombinedEmbedDim]

        # 7. Decode
        recon_logits = self.decode(decoder_input)
        _ = F.kl_div(F.log_softmax(recon_logits, dim=-1), F.log_softmax(attn_weights, dim=-1),
                       reduction='batchmean', log_target=True)

        # 8. Predict Static Distribution (from mu)
        predicted_static_dist_logits = self.static_pred_head(mu)

        # Return same signature as before for compatibility with train/analysis
        return recon_logits, mu, logvar, predicted_static_dist_logits
