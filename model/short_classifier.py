import torch
import torch.nn as nn
import torch.nn.functional as F
from model.kronos import Kronos, KronosTokenizer


class KronosShortClassifier(nn.Module):
    """
    KronosShortClassifier module for detecting blow-off tops and short trade signals.

    Combines Kronos Transformer encoder backbone (as a feature extractor)
    with a multi-layer perceptron (MLP) classification head.
    """

    def __init__(self, 
                 kronos_model: Kronos, 
                 d_model: int = 256, 
                 classifier_hidden_dim: int = 128, 
                 dropout: float = 0.2,
                 freeze_backbone: bool = False):
        super().__init__()
        self.kronos = kronos_model
        self.d_model = d_model

        if freeze_backbone:
            for param in self.kronos.parameters():
                param.requires_grad = False

        # Multi-layer classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1)
        )

    def extract_features(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor, stamp: torch.Tensor = None, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes tokens through Kronos Transformer layers and extracts final timestep context vector.
        
        Args:
            s1_ids: Shape [batch_size, seq_len]
            s2_ids: Shape [batch_size, seq_len]
            stamp: Shape [batch_size, seq_len, time_feats]
        Returns:
            torch.Tensor: Feature embedding of final timestep, Shape [batch_size, d_model]
        """
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.kronos.time_emb(stamp)
            x = x + time_embedding
        x = self.kronos.token_drop(x)

        for layer in self.kronos.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.kronos.norm(x)
        # Extract representation at the final historical candle (last sequence step)
        final_token_embedding = x[:, -1, :]
        return final_token_embedding

    def forward(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor, stamp: torch.Tensor = None, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass returning unscaled classification logits.
        
        Returns:
            torch.Tensor: Logits of shape [batch_size, 1]
        """
        features = self.extract_features(s1_ids, s2_ids, stamp=stamp, padding_mask=padding_mask)
        logits = self.classifier(features)
        return logits

    def predict_probability(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor, stamp: torch.Tensor = None) -> torch.Tensor:
        """
        Inference method returning Sigmoid probability scores in range [0, 1].
        
        Returns:
            torch.Tensor: Probabilities of shape [batch_size, 1]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(s1_ids, s2_ids, stamp=stamp)
            probs = torch.sigmoid(logits)
        return probs
