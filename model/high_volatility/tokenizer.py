from model.kronos import KronosTokenizer


class HighVolatilityTokenizer(KronosTokenizer):
    """High-volatility variant of KronosTokenizer.

    No architectural changes -- Area A/B operate on the raw window before/
    after tokenization, not inside the tokenizer itself. Kept as a distinct
    subclass so the high-volatility model line has its own class/checkpoint
    identity, independent from the base KronosTokenizer.
    """
    pass
