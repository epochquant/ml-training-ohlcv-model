import torch
import torch.nn.functional as F

from model.kronos import Kronos
from model.high_volatility.regime import RegimeEmbedding


class HighVolatilityKronos(Kronos):
    """High-volatility variant of Kronos with an additional zero-initialized
    regime embedding (Area B / B1) that injects a continuous multi-scale
    volatility-regime signal (ATR%, short & macro overextension, multi-timeframe
    returns, volume-ratio, taker-ratio, wick rejection -- see model/high_volatility/regime.py)
    alongside the existing temporal embedding.

    Zero-initialization means this is a true no-op immediately after
    warm-starting from a pretrained base Kronos checkpoint -- it only starts
    contributing once fine-tuned.
    """

    def __init__(self, *args, n_regime_features: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_regime_features = n_regime_features
        self.regime_emb = RegimeEmbedding(n_regime_features, self.d_model)

    def _embed(self, s1_ids, s2_ids, stamp=None, regime=None, padding_mask=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if regime is not None:
            x = x + self.regime_emb(regime)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        return x

    def forward(self, s1_ids, s2_ids, stamp=None, regime=None, padding_mask=None,
                use_teacher_forcing=False, s1_targets=None):
        x = self._embed(s1_ids, s2_ids, stamp=stamp, regime=regime, padding_mask=padding_mask)
        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, regime=None, padding_mask=None):
        x = self._embed(s1_ids, s2_ids, stamp=stamp, regime=regime, padding_mask=padding_mask)
        s1_logits = self.head(x)
        return s1_logits, x


def load_hv_kronos_from_base(pretrained_path: str, n_regime_features: int = 8) -> "HighVolatilityKronos":
    """Warm-start a HighVolatilityKronos from a base Kronos checkpoint.

    Does not use HighVolatilityKronos.from_pretrained directly -- the
    checkpoint's config.json only has the base Kronos constructor args, and
    the base PyTorchModelHubMixin.from_pretrained state-dict load is strict
    by default, which would fail on the extra regime_emb.* parameters. Instead,
    the base model's architecture args are read off the loaded instance and
    its weights are loaded into the new instance with strict=False, so only
    regime_emb.* (zero-initialized, see RegimeEmbedding) is left uninitialized.
    """
    base_model = Kronos.from_pretrained(pretrained_path)
    hv_model = HighVolatilityKronos(
        s1_bits=base_model.s1_bits,
        s2_bits=base_model.s2_bits,
        n_layers=base_model.n_layers,
        d_model=base_model.d_model,
        n_heads=base_model.n_heads,
        ff_dim=base_model.ff_dim,
        ffn_dropout_p=base_model.ffn_dropout_p,
        attn_dropout_p=base_model.attn_dropout_p,
        resid_dropout_p=base_model.resid_dropout_p,
        token_dropout_p=base_model.token_dropout_p,
        learn_te=base_model.learn_te,
        n_regime_features=n_regime_features,
    )
    missing, unexpected = hv_model.load_state_dict(base_model.state_dict(), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when warm-starting HighVolatilityKronos: {unexpected}")
    if any("regime_emb" not in k for k in missing):
        raise RuntimeError(f"Unexpected missing keys when warm-starting HighVolatilityKronos: {missing}")
    return hv_model
