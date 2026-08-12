import torch
import torch.nn as nn

from model.short_classifier import KronosShortClassifier


class HighVolatilityShortClassifier(KronosShortClassifier):
    """High-volatility variant of KronosShortClassifier (Area C).

    Fuses volatility-regime features (e.g. ATR%, overextension, vol_ratio,
    upper_wick_ratio -- see src/data/label_short_signals_hv.py) with the
    Kronos backbone's final-timestep embedding before the classifier head,
    instead of relying only on the embedding to implicitly encode them.
    """

    def __init__(self, kronos_model, d_model: int = 256, classifier_hidden_dim: int = 128,
                 dropout: float = 0.2, freeze_backbone: bool = False, n_extra_features: int = 4):
        super().__init__(kronos_model, d_model=d_model, classifier_hidden_dim=classifier_hidden_dim,
                          dropout=dropout, freeze_backbone=freeze_backbone)
        self.n_extra_features = n_extra_features
        fused_dim = d_model + n_extra_features
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1),
        )

    def forward(self, s1_ids, s2_ids, extra_features, stamp=None, padding_mask=None):
        features = self.extract_features(s1_ids, s2_ids, stamp=stamp, padding_mask=padding_mask)
        fused = torch.cat([features, extra_features], dim=-1)
        return self.classifier(fused)

    def predict_probability(self, s1_ids, s2_ids, extra_features, stamp=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(s1_ids, s2_ids, extra_features, stamp=stamp)
            probs = torch.sigmoid(logits)
        return probs
