import numpy as np
import pandas as pd
import torch

from model.kronos import KronosPredictor, sample_from_logits, calc_time_stamps
from model.high_volatility.normalization import normalize_window, denormalize_continuation
from model.high_volatility.regime import compute_regime_vector


def hv_auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context,
                                  pred_len, clip=5, T=1.0, top_k=0, top_p=0.99,
                                  sample_count=5, regime=None, verbose=False):
    """Same rolling-buffer autoregressive loop as model.kronos.auto_regressive_inference,
    but (a) optionally injects a per-window regime vector (Area B) into every decode
    step, and (b) returns the un-reduced per-sample ensemble instead of collapsing it
    to a mean (Area D) -- callers decide how to reduce (mean, quantiles, ...).
    """
    with torch.no_grad():
        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)
        if regime is not None:
            regime = regime.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, regime.size(1), regime.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        full_regime = None
        if regime is not None:
            full_regime = regime[:, -1:, :].repeat(1, total_seq_len, 1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
            current_regime = None
            if full_regime is not None:
                current_regime = full_regime[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], stamp=current_stamp, regime=current_regime)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

            if verbose:
                print(f"step {i + 1}/{pred_len}")

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        return preds


class HighVolatilityPredictor(KronosPredictor):
    """High-volatility variant of KronosPredictor.

    Combines Area A (log-return/robust normalization + soft clip, replacing the
    base predictor's raw-price z-score + hard clip), Area D (returns the raw
    per-sample ensemble so callers can request quantiles instead of only a
    mean), and Area B (computes and forwards a per-window regime vector).
    """

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5.0,
                 normalization_mode="logreturn", soft_clip=True, use_regime=True):
        super().__init__(model, tokenizer, device=device, max_context=max_context, clip=clip)
        self.normalization_mode = normalization_mode
        self.soft_clip = soft_clip
        self.use_regime = use_regime

    def _prepare_regime(self, x_raw_batch):
        regime_batch = np.stack([compute_regime_vector(x_raw) for x_raw in x_raw_batch], axis=0)
        return regime_batch

    def generate_with_quantiles(self, x, x_stamp, y_stamp, pred_len, T=1.0, top_k=0,
                                 top_p=0.99, sample_count=20, regime=None, verbose=False):
        x_t = torch.from_numpy(x).float().to(self.device)
        x_stamp_t = torch.from_numpy(x_stamp).float().to(self.device)
        y_stamp_t = torch.from_numpy(y_stamp).float().to(self.device)
        regime_t = None
        if regime is not None:
            regime_t = torch.from_numpy(regime).float().to(self.device)

        preds = hv_auto_regressive_inference(
            self.tokenizer, self.model, x_t, x_stamp_t, y_stamp_t, self.max_context,
            pred_len, clip=self.clip, T=T, top_k=top_k, top_p=top_p,
            sample_count=sample_count, regime=regime_t, verbose=verbose,
        )
        return preds[:, :, -pred_len:, :]

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9,
                sample_count=20, verbose=True, return_quantiles=False, quantiles=(0.1, 0.5, 0.9)):
        results = self.predict_batch(
            [df], [x_timestamp], [y_timestamp], pred_len, T=T, top_k=top_k, top_p=top_p,
            sample_count=sample_count, verbose=verbose, return_quantiles=return_quantiles,
            quantiles=quantiles,
        )
        return results[0]

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0,
                       top_k=0, top_p=0.9, sample_count=20, verbose=True,
                       return_quantiles=False, quantiles=(0.1, 0.5, 0.9)):
        price_cols = ['open', 'high', 'low', 'close']
        vol_col, amt_col = 'volume', 'amount'

        df_list_filled = []
        for df in df_list:
            df = df.copy()
            for col in price_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' in input DataFrame.")
            if vol_col not in df.columns:
                df[vol_col] = 0.0
                df[amt_col] = 0.0
            if amt_col not in df.columns and vol_col in df.columns:
                df[amt_col] = df[vol_col] * df[price_cols].mean(axis=1)
            if df[price_cols + [vol_col, amt_col]].isnull().values.any():
                raise ValueError("Input DataFrame contains NaN values in price or volume columns.")
            df_list_filled.append(df)

        feature_cols = price_cols + [vol_col, amt_col]
        x_raw_batch = [df[feature_cols].values.astype(np.float64) for df in df_list_filled]

        x_norm_list, stats_list = [], []
        for x_raw in x_raw_batch:
            x_norm, stats = normalize_window(x_raw, mode=self.normalization_mode, clip=self.clip,
                                              soft_clip=self.soft_clip)
            x_norm_list.append(x_norm)
            stats_list.append(stats)
        x_norm_batch = np.stack(x_norm_list, axis=0)

        regime_batch = None
        if self.use_regime:
            regime_vec = self._prepare_regime(x_raw_batch)
            seq_len = x_norm_batch.shape[1]
            regime_batch = np.repeat(regime_vec[:, None, :], seq_len, axis=1)

        x_stamp_batch = np.stack(
            [calc_time_stamps(ts).values.astype(np.float32) for ts in x_timestamp_list], axis=0
        )
        y_stamp_batch = np.stack(
            [calc_time_stamps(ts).values.astype(np.float32) for ts in y_timestamp_list], axis=0
        )

        preds = self.generate_with_quantiles(
            x_norm_batch, x_stamp_batch, y_stamp_batch, pred_len, T=T, top_k=top_k, top_p=top_p,
            sample_count=sample_count, regime=regime_batch, verbose=verbose,
        )

        results = []
        for b, stats in enumerate(stats_list):
            denorm_samples = np.stack(
                [denormalize_continuation(preds[b, s], stats) for s in range(preds.shape[1])], axis=0
            )
            y_index = y_timestamp_list[b]
            if return_quantiles:
                q_arr = np.quantile(denorm_samples, quantiles, axis=0)
                out = {}
                for qi, q in enumerate(quantiles):
                    label = f"p{int(round(q * 100))}"
                    out[label] = pd.DataFrame(q_arr[qi], columns=feature_cols, index=y_index)
                results.append(out)
            else:
                mean_arr = denorm_samples.mean(axis=0)
                results.append(pd.DataFrame(mean_arr, columns=feature_cols, index=y_index))

        return results
