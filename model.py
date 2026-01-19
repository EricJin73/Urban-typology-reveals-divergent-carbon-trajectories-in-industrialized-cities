
"""
FiLM Seq2Seq (v17 clean) + Monte Carlo Dropout Uncertainty

Created: 2025-11-04 (JST)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# === Configuration ===
CITY_DATA_PATH = r"input path"
SSP_DATA_PATH  = r"ssp_path"

# Countries to run
COUNTRY_CODES  = [
                  'AUS', 'AUT', 'BEL', 'BGR', 'BLR', 'CAN', 'CHE', 'CYP', 'CZE',
                  'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV',
                  'HUN', 'IRL', 'ISL', 'ITA', 'JPN', 'LIE', 'LTU', 'LUX', 'LVA',
                  'MCO', 'MLT', 'NLD', 'NOR', 'NZL', 'POL', 'PRT', 'ROU', 'RUS',
                  'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'USA'
                  ]

SCENARIOS      = ['BAU', 'SSP1', 'SSP5']

HIST_START, HIST_END     = 2001, 2020
FUTURE_START, FUTURE_END = 2021, 2050

VARIABLES      = ['gdp', 'pop', 'builtup_area', 'cropland_area', 'forest_area']  # adjust names to your data

SEQ_LEN        = 12            # encoder sequence length (↑ helps trend)
PRED_LEN_TRAIN = 4             # decoder steps during training (multi-step)
BATCH_SIZE     = 128
EPOCHS         = 50
LR             = 8e-4
WEIGHT_DECAY   = 1e-5
EMB_DIM        = 12
ENC_HIDDEN     = 128
DEC_HIDDEN     = 128
NUM_LAYERS     = 2
DROPOUT        = 0.1
CLIP_NORM      = 1.0          # gradient clipping (for stability)

# === Monte Carlo (MC) Dropout settings ===
USE_MC         = True          # turn on/off MC uncertainty export for scenarios
MC_SAMPLES     = 20            # number of stochastic forward passes per city-scenario
MC_Q_LO        = 0.05          # lower quantile (e.g., 5%)
MC_Q_HI        = 0.95          # upper quantile (e.g., 95%)

DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED    = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# === Output dirs ===
OUT_DIR_ROOT   = "output"
OUT_DIR_CITY   = os.path.join(OUT_DIR_ROOT, "city")
OUT_DIR_SUMMARY= os.path.join(OUT_DIR_ROOT, "summary")
OUT_DIR_LOSS   = os.path.join(OUT_DIR_ROOT, "loss")
os.makedirs(OUT_DIR_CITY, exist_ok=True)
os.makedirs(OUT_DIR_SUMMARY, exist_ok=True)
os.makedirs(OUT_DIR_LOSS, exist_ok=True)

# === Utilities ===
# === Linear extrapolation for BAU ===
def extrapolate_variables_bau(df, vars_list, hist_start, hist_end, future_start, future_end):
    """
    Linear extrapolation for BAU with non-negativity.
    """
    df_hist = df[(df['year'] >= hist_start) & (df['year'] <= hist_end)].copy()
    if df_hist.empty:
        return None
    df_hist['t'] = df_hist['year'] - hist_start
    years_f = np.arange(future_start, future_end + 1)
    df_fut = pd.DataFrame({'year': years_f})

    for var in vars_list:
        if df_hist[var].isna().all():
            df_fut[var] = 0.0
            continue
        y = df_hist[var].values
        model = sm.OLS(y, sm.add_constant(df_hist['t'])).fit()
        predicted = model.predict(sm.add_constant(years_f - hist_start))
        df_fut[var] = np.maximum(predicted, 0)
    return df_fut

# === SSP-based extrapolation (hybrid) ===
def extrapolate_with_ssp(df, vars_list, hist_start, hist_end, future_start, future_end, ssp_pivot, scenario):
    df_histwin = df[(df['year'] >= hist_start) & (df['year'] <= hist_end)]
    if df_histwin.empty or hist_end not in df['year'].values:
        return None
    hist_vals = df[df['year'] == hist_end][vars_list].iloc[0]
    if scenario not in ssp_pivot:
        return None
    ssp_df = ssp_pivot[scenario]
    if FUTURE_START < ssp_df.index.min() or FUTURE_END > ssp_df.index.max():
        ssp_df = ssp_df.reindex(np.arange(min(ssp_df.index.min(), FUTURE_START),
                                          max(ssp_df.index.max(), FUTURE_END) + 1)).interpolate().ffill().bfill()
    ssp_future = ssp_df.loc[future_start:future_end]
    years_f = np.arange(future_start, future_end + 1)
    df_fut = pd.DataFrame({'year': years_f})

    df_hist = df_histwin.copy()
    df_hist['t'] = df_hist['year'] - hist_start

    for var in vars_list:
        if var in ssp_df.columns and hist_end in ssp_df.index:
            base = ssp_df.at[hist_end, var]
            if base == 0:
                base = 1e-10
            ratio = ssp_future[var].values / base
            ssp_values = hist_vals[var] * ratio
        else:
            ssp_values = None

        if df_hist[var].isna().all():
            linear_values = np.zeros_like(years_f, dtype=float)
        else:
            y = df_hist[var].values
            model = sm.OLS(y, sm.add_constant(df_hist['t'])).fit()
            linear_values = model.predict(sm.add_constant(years_f - hist_start))

        hybrid_values = linear_values if ssp_values is None else 0.5 * ssp_values + 0.5 * linear_values
        df_fut[var] = np.maximum(hybrid_values, 0)
    return df_fut

# === Sequence builder (multi-step) ===
def build_city_sequences(df, city_idx, pred_len):
    """
    For a single city:
      Returns:
        X_enc  : [N, SEQ_LEN, F]
        years_enc : [N, SEQ_LEN] (not used but kept for compatibility)
        X_dec  : [N, pred_len, F]
        years_dec : [N, pred_len]
        y      : [N, pred_len]
        idx    : [N]
    """
    df_h = df[(df['year'] >= HIST_START) & (df['year'] <= HIST_END)].sort_values('year')
    X = df_h[VARIABLES].values.astype(np.float32)
    y = df_h['co2'].values.astype(np.float32)
    years = df_h['year'].values
    X_enc, Y, X_dec, idx, years_enc_list, years_dec_list = [], [], [], [], [], []
    for i in range(len(X) - SEQ_LEN - pred_len + 1):
        enc = X[i:i + SEQ_LEN]
        dec = X[i + SEQ_LEN: i + SEQ_LEN + pred_len]
        tgt = y[i + SEQ_LEN: i + SEQ_LEN + pred_len]
        yrs_enc = years[i:i + SEQ_LEN]
        yrs_dec = years[i + SEQ_LEN: i + SEQ_LEN + pred_len]

        X_enc.append(enc)
        X_dec.append(dec)
        Y.append(tgt)
        years_enc_list.append(yrs_enc)
        years_dec_list.append(yrs_dec)
        idx.append(city_idx)
    if not X_enc:
        return (np.zeros((0, SEQ_LEN, len(VARIABLES)), np.float32),
                np.zeros((0, SEQ_LEN), np.int32),
                np.zeros((0, pred_len, len(VARIABLES)), np.float32),
                np.zeros((0, pred_len), np.int32),
                np.zeros((0, pred_len), np.float32),
                np.zeros((0,), np.int64))
    return (np.array(X_enc), np.array(years_enc_list, dtype=np.int32),
            np.array(X_dec), np.array(years_dec_list, dtype=np.int32),
            np.array(Y), np.array(idx, dtype=np.int64))

# === Model ===
class FiLMSeq2Seq(nn.Module):
    def __init__(self, n_cities, emb_dim, feat_dim,
                 enc_hidden=64, dec_hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.feat_dim  = feat_dim
        self.enc_hidden= enc_hidden
        self.dec_hidden= dec_hidden
        self.num_layers= num_layers

        self.city_emb = nn.Embedding(n_cities, emb_dim)
        self.film_gen = nn.Linear(emb_dim, 2 * feat_dim)  # gamma, beta

        self.encoder = nn.LSTM(feat_dim, enc_hidden, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(feat_dim, dec_hidden, num_layers, batch_first=True, dropout=dropout)

        # ---- Attention (Luong-general) ----
        self.use_attention = True
        self.enc_out_proj = nn.Linear(enc_hidden, dec_hidden) if enc_hidden != dec_hidden else nn.Identity()
        self.attn_in_proj = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.head_attn = nn.Sequential(
            nn.Linear(dec_hidden * 2, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, 1)
        )
        # Fallback head (no attention)
        self.head = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden // 2, 1)
        )

        if enc_hidden != dec_hidden:
            self.bridge_h = nn.Linear(enc_hidden, dec_hidden)
            self.bridge_c = nn.Linear(enc_hidden, dec_hidden)
        else:
            self.bridge_h = self.bridge_c = None

    def apply_film(self, x, city_idx):
        emb = self.city_emb(city_idx)             # [B, emb_dim]
        gamma, beta = self.film_gen(emb).chunk(2, dim=1)  # [B, feat_dim] each
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

    def encode(self, x_enc, city_idx):
        x_mod = self.apply_film(x_enc, city_idx)
        enc_out, (h, c) = self.encoder(x_mod)           # [B, L, H_enc]
        if self.bridge_h is not None:
            h = self.bridge_h(h.transpose(0,1)).transpose(0,1)
            c = self.bridge_c(c.transpose(0,1)).transpose(0,1)
        enc_ctx = self.enc_out_proj(enc_out)            # [B, L, H_dec]
        return (h.contiguous(), c.contiguous()), enc_ctx

    def decode(self, x_dec, hc, city_idx, enc_ctx):
        x_mod = self.apply_film(x_dec, city_idx)
        dec_out, _ = self.decoder(x_mod, hc)            # [B, T, H_dec]

        if self.use_attention:
            q = self.attn_in_proj(dec_out)              # [B, T, H_dec]
            attn_scores = torch.bmm(q, enc_ctx.transpose(1, 2))  # [B, T, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)    # [B, T, L]
            context = torch.bmm(attn_weights, enc_ctx)           # [B, T, H_dec]
            fused = torch.cat([dec_out, context], dim=-1)        # [B, T, 2H_dec]
            y = self.head_attn(fused).squeeze(-1)                # [B, T]
        else:
            y = self.head(dec_out).squeeze(-1)                   # [B, T]
        return y

    def forward(self, x_enc, city_idx, x_dec):
        hc0, enc_ctx = self.encode(x_enc, city_idx)
        y = self.decode(x_dec, hc0, city_idx, enc_ctx)
        return y


# === Metrics helpers ===
def flatten_stepwise(arr_2d):
    return arr_2d.reshape(-1)

def calculate_metrics_seq2seq(model, loader, device, sy_list=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb_enc, years_enc, xb_dec, years_dec, city_idx, yb in loader:
            xb_enc = xb_enc.to(device)
            xb_dec = xb_dec.to(device)
            city_idx = city_idx.to(device)
            yb = yb.to(device)                       # [B, T]
            pred = model(xb_enc, city_idx, xb_dec)   # [B, T]

            if sy_list is None:
                y_true.extend(flatten_stepwise(yb.cpu().numpy()))
                y_pred.extend(flatten_stepwise(pred.cpu().numpy()))
            else:
                pred_np = pred.cpu().numpy()
                yb_np   = yb.cpu().numpy()
                city_np = city_idx.cpu().numpy()
                B, T = pred_np.shape
                for b in range(B):
                    c = int(city_np[b])
                    scaler = sy_list[c]
                    if scaler is None:
                        continue
                    y_pred.extend(scaler.inverse_transform(pred_np[b].reshape(-1,1)).reshape(-1))
                    y_true.extend(scaler.inverse_transform(yb_np[b].reshape(-1,1)).reshape(-1))

    if len(y_true) == 0:
        return np.nan, np.nan
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

def predict_mu_seq2seq(model, Xenc_np, Xdec_np, city_idx_np, device, batch_size=512):
    model.eval()
    outs = []
    with torch.no_grad():
        n = len(Xenc_np)
        for i in range(0, n, batch_size):
            xb_enc = torch.from_numpy(Xenc_np[i:i+batch_size].astype(np.float32)).to(device)
            xb_dec = torch.from_numpy(Xdec_np[i:i+batch_size].astype(np.float32)).to(device)
            cb     = torch.from_numpy(city_idx_np[i:i+batch_size].astype(np.int64)).to(device)
            pred = model(xb_enc, cb, xb_dec)   # [B, T]
            outs.append(pred.cpu().numpy())
    if not outs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outs, axis=0)  # [N, T]

def mc_decode_samples(model, x_enc_t, x_dec_t, cidx_t, n_samples):
    """
    Run n stochastic forward passes with dropout enabled.
    Returns: np.ndarray of shape [n_samples, T] (normalized space).
    """
    preds = []
    model.train()  # enable dropout in LSTMs & heads
    with torch.no_grad():
        for _ in range(n_samples):
            y_norm = model(x_enc_t, cidx_t, x_dec_t)  # [1, T]
            preds.append(y_norm.squeeze(0).cpu().numpy())
    model.eval()
    return np.stack(preds, axis=0)  # [S, T]

# === Core training per country ===
def train_and_forecast_one_country(df_all, ssp_pivot, country_code):
    df_single = df_all[df_all['GID_0'] == country_code]
    if df_single.empty:
        return None, {"country": country_code, "reason": "no_rows_in_city_dataset"}, None

    cities = df_single['UID'].unique()
    if len(cities) == 0:
        return None, {"country": country_code, "reason": "no_cities_found"}, None

    city2idx = {city: i for i, city in enumerate(cities)}
    idx2city = {i: c for c, i in city2idx.items()}

    per_city = {}
    for city in cities:
        df_c = df_single[df_single['UID'] == city].sort_values('year')
        if df_c['year'].nunique() < SEQ_LEN + PRED_LEN_TRAIN or df_c[['co2'] + VARIABLES].isna().any().any():
            continue

        Xenc, Yenc, Xdec, Ydec, Y, idx = build_city_sequences(df_c, city2idx[city], PRED_LEN_TRAIN)
        if len(Xenc) == 0: 
            continue

        n_samples = len(Xenc)
        train_size = max(1, int(0.8 * n_samples))
        perm = np.random.permutation(n_samples)
        tr, va = perm[:train_size], perm[train_size:]

        per_city[city] = {
            "Xenc_tr": Xenc[tr], "Yenc_tr": Yenc[tr],
            "Xdec_tr": Xdec[tr], "Ydec_tr": Ydec[tr],
            "Y_tr": Y[tr],
            "Xenc_va": Xenc[va], "Yenc_va": Yenc[va],
            "Xdec_va": Xdec[va], "Ydec_va": Ydec[va],
            "Y_va": Y[va]
        }

    if not per_city:
        return None, {"country": country_code, "reason": "no_cities_with_enough_history"}, None

    # Fit scalers (per city, on encoder+decoder)
    city_x_scalers, city_y_scalers = {}, {}
    for city, d in per_city.items():
        if len(d["Xenc_tr"]) == 0: 
            continue
        enc_flat = d["Xenc_tr"].reshape(-1, len(VARIABLES))
        dec_flat = d["Xdec_tr"].reshape(-1, len(VARIABLES))
        sx = StandardScaler().fit(np.vstack([enc_flat, dec_flat]))
        sy = StandardScaler().fit(d["Y_tr"].reshape(-1, 1))
        city_x_scalers[city] = sx
        city_y_scalers[city] = sy

    # Compose scale + concat city-wise
    def scale_pair(Xenc, Xdec, city):
        sx = city_x_scalers[city]
        f_nope = len(VARIABLES)
        enc_core = Xenc.reshape(-1, f_nope)
        dec_core = Xdec.reshape(-1, f_nope)
        enc_core_n = sx.transform(enc_core).reshape(Xenc.shape[0], Xenc.shape[1], f_nope)
        dec_core_n = sx.transform(dec_core).reshape(Xdec.shape[0], Xdec.shape[1], f_nope)
        return enc_core_n.astype(np.float32), dec_core_n.astype(np.float32)

    Xenc_tr_list, Xdec_tr_list, y_tr_list, city_idx_tr_list, years_tr_list = [], [], [], [], []
    Xenc_va_list, Xdec_va_list, y_va_list, city_idx_va_list, years_va_list = [], [], [], [], []

    for city, d in per_city.items():
        if city not in city_x_scalers: 
            continue
        Xenc_tr, Xdec_tr = scale_pair(d["Xenc_tr"], d["Xdec_tr"], city)
        Xenc_va, Xdec_va = scale_pair(d["Xenc_va"], d["Xdec_va"], city)
        sy = city_y_scalers[city]
        y_tr = sy.transform(d["Y_tr"].reshape(-1,1)).reshape(d["Y_tr"].shape)
        y_va = sy.transform(d["Y_va"].reshape(-1,1)).reshape(d["Y_va"].shape)

        Xenc_tr_list.append(Xenc_tr); Xdec_tr_list.append(Xdec_tr); y_tr_list.append(y_tr)
        Xenc_va_list.append(Xenc_va); Xdec_va_list.append(Xdec_va); y_va_list.append(y_va)

        city_idx_tr_list.append(np.full(len(Xenc_tr), city2idx[city], dtype=np.int64))
        city_idx_va_list.append(np.full(len(Xenc_va), city2idx[city], dtype=np.int64))

        years_tr_list.append(d["Ydec_tr"])  # target years
        years_va_list.append(d["Ydec_va"])

    if not Xenc_tr_list:
        return None, {"country": country_code, "reason": "failed_to_prepare_training_data"}, None

    Xenc_train = np.concatenate(Xenc_tr_list, axis=0)
    Xdec_train = np.concatenate(Xdec_tr_list, axis=0)
    y_train    = np.concatenate(y_tr_list, axis=0)
    city_idx_train = np.concatenate(city_idx_tr_list, axis=0)
    years_train    = np.concatenate(years_tr_list, axis=0)

    if Xenc_va_list:
        Xenc_val = np.concatenate(Xenc_va_list, axis=0)
        Xdec_val = np.concatenate(Xdec_va_list, axis=0)
        y_val    = np.concatenate(y_va_list, axis=0)
        city_idx_val = np.concatenate(city_idx_va_list, axis=0)
        years_val    = np.concatenate(years_va_list, axis=0)
    else:
        n = len(Xenc_train)
        val_size = max(1, n // 10)
        Xenc_val, Xdec_val = Xenc_train[-val_size:], Xdec_train[-val_size:]
        y_val, city_idx_val, years_val = y_train[-val_size:], city_idx_train[-val_size:], years_train[-val_size:]
        Xenc_train, Xdec_train = Xenc_train[:-val_size], Xdec_train[:-val_size]
        y_train, city_idx_train, years_train = y_train[:-val_size], city_idx_train[:-val_size], years_train[:-val_size]

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(Xenc_train.astype(np.float32)),
        torch.from_numpy(years_train.astype(np.int32)),
        torch.from_numpy(Xdec_train.astype(np.float32)),
        torch.from_numpy(years_train.astype(np.int32)),
        torch.from_numpy(city_idx_train.astype(np.int64)),
        torch.from_numpy(y_train.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.from_numpy(Xenc_val.astype(np.float32)),
        torch.from_numpy(years_val.astype(np.int32)),
        torch.from_numpy(Xdec_val.astype(np.float32)),
        torch.from_numpy(years_val.astype(np.int32)),
        torch.from_numpy(city_idx_val.astype(np.int64)),
        torch.from_numpy(y_val.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # y scalers list
    sy_list = [None] * len(city2idx)
    for city, idx in city2idx.items():
        if city in city_y_scalers:
            sy_list[idx] = city_y_scalers[city]

    FEAT_DIM = len(VARIABLES)
    model = FiLMSeq2Seq(
        n_cities=len(city2idx),
        emb_dim=EMB_DIM,
        feat_dim=FEAT_DIM,
        enc_hidden=ENC_HIDDEN,
        dec_hidden=DEC_HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val = float('inf'); best_state = None
    train_hist, val_hist = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        loss_sum = 0.0; n_sum = 0
        for xb_enc, yenc, xb_dec, ydec, cb, yb in train_loader:
            xb_enc = xb_enc.to(DEVICE); xb_dec = xb_dec.to(DEVICE)
            cb = cb.to(DEVICE); yb = yb.to(DEVICE)  # [B, T]
            pred = model(xb_enc, cb, xb_dec)        # [B, T]
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            if CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            bs = xb_enc.size(0)
            loss_sum += loss.item() * bs; n_sum += bs
        train_loss = loss_sum / max(1, n_sum)

        model.eval()
        val_sum = 0.0; vn = 0
        with torch.no_grad():
            for xb_enc, yenc, xb_dec, ydec, cb, yb in val_loader:
                xb_enc = xb_enc.to(DEVICE); xb_dec = xb_dec.to(DEVICE)
                cb = cb.to(DEVICE); yb = yb.to(DEVICE)
                pred = model(xb_enc, cb, xb_dec)
                vloss = criterion(pred, yb)
                bs = xb_enc.size(0)
                val_sum += vloss.item() * bs; vn += bs
        val_loss = val_sum / max(1, vn)
        train_hist.append(train_loss); val_hist.append(val_loss)

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            print(f"[{country_code}] Ep{epoch:03d} tr{train_loss:.4f} va{val_loss:.4f} *")
        elif epoch % 10 == 0:
            print(f"[{country_code}] Ep{epoch:03d} tr{train_loss:.4f} va{val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save loss curves
    pd.DataFrame({"epoch": np.arange(1, len(train_hist)+1), "mse": train_hist}).to_csv(
        os.path.join(OUT_DIR_LOSS, f"{country_code.lower()}_train_loss_v17_mc.csv"), index=False)
    pd.DataFrame({"epoch": np.arange(1, len(val_hist)+1), "mse": val_hist}).to_csv(
        os.path.join(OUT_DIR_LOSS, f"{country_code.lower()}_val_loss_v17_mc.csv"), index=False)

    # Metrics (inverse)
    tr_r2, tr_rmse = calculate_metrics_seq2seq(model, train_loader, DEVICE, sy_list)
    va_r2, va_rmse = calculate_metrics_seq2seq(model, val_loader, DEVICE, sy_list)

    # Export train/val predictions (denorm)
    mu_tr = predict_mu_seq2seq(model, Xenc_train, Xdec_train, city_idx_train, DEVICE, batch_size=max(256,BATCH_SIZE))  # [N,T]
    mu_va = predict_mu_seq2seq(model, Xenc_val,   Xdec_val,   city_idx_val,   DEVICE, batch_size=max(256,BATCH_SIZE))  # [N,T]

    def inv_block(mu_norm, y_norm, cidx_vec, years_mat):
        recs = []
        for i in range(mu_norm.shape[0]):
            c = int(cidx_vec[i]); scaler = sy_list[c]
            years = years_mat[i]
            mu_d = scaler.inverse_transform(mu_norm[i].reshape(-1,1)).reshape(-1)
            y_d  = scaler.inverse_transform(y_norm[i].reshape(-1,1)).reshape(-1)
            uid  = idx2city[c]
            for t, year in enumerate(years):
                recs.append((uid, country_code, int(year), float(y_d[t]), float(mu_d[t])))
        return recs

    tv_recs = []
    tv_recs += inv_block(mu_tr, y_train, city_idx_train, years_train)
    tv_recs += inv_block(mu_va, y_val,   city_idx_val,    years_val)  # correct: pass city_idx_val

    df_tv = pd.DataFrame(tv_recs, columns=["UID","GID_0","year","co2_true","co2_pred"])
    df_tv['split'] = 'train_or_val'
    df_tv.to_csv(os.path.join(OUT_DIR_CITY, f"{country_code.lower()}_trainval_pred_v19.csv"), index=False)

    # === Scenario forecasts with full-horizon decoding ===
    all_results = []
    for scenario in SCENARIOS:
        scenario_results = []
        for city, cidx in city2idx.items():
            df_c = df_single[df_single['UID'] == city].sort_values('year')
            if scenario == 'BAU':
                df_fut = extrapolate_variables_bau(df_c, VARIABLES, HIST_START, HIST_END, FUTURE_START, FUTURE_END)
            else:
                df_fut = extrapolate_with_ssp(df_c, VARIABLES, HIST_START, HIST_END, FUTURE_START, FUTURE_END, ssp_pivot, scenario)
            if df_fut is None:
                continue
            # encoder window
            hist_feats = df_c[df_c['year'] <= HIST_END][VARIABLES].values.astype(np.float32)
            hist_years = df_c[df_c['year'] <= HIST_END]['year'].values
            if len(hist_feats) < SEQ_LEN:
                continue
            enc_window = hist_feats[-SEQ_LEN:]        # [L,F]
            enc_input = enc_window[None, ...]         # [1,L,F]

            dec_feats = df_fut[VARIABLES].values.astype(np.float32)      # [T,F]
            dec_input = dec_feats[None, ...]          # [1,T,F]

            sx = city_x_scalers.get(city, None); sy = city_y_scalers.get(city, None)
            if sx is None or sy is None: 
                continue
            f_nope = len(VARIABLES)
            enc_core = enc_input[...,:f_nope].reshape(-1, f_nope)
            enc_core_n = sx.transform(enc_core).reshape(1, SEQ_LEN, f_nope)
            dec_core = dec_input[...,:f_nope].reshape(-1, f_nope)
            dec_core_n = sx.transform(dec_core).reshape(1, dec_input.shape[1], f_nope)
            x_enc = enc_core_n.astype(np.float32)
            x_dec = dec_core_n.astype(np.float32)

            # Deterministic point estimate
            with torch.no_grad():
                y_norm = model(torch.from_numpy(x_enc).to(DEVICE),
                               torch.tensor([cidx], dtype=torch.long, device=DEVICE),
                               torch.from_numpy(x_dec).to(DEVICE)).cpu().numpy().reshape(-1)  # [T]
            y_den = sy.inverse_transform(y_norm.reshape(-1,1)).reshape(-1)

            out = df_fut.copy()
            out['co2_pred'] = y_den
            out['UID'] = city; out['GID_0'] = country_code; out['scenario'] = scenario

            # MC Dropout statistics
            if USE_MC:
                x_enc_t = torch.from_numpy(x_enc).float().to(DEVICE)
                x_dec_t = torch.from_numpy(x_dec).float().to(DEVICE)
                cidx_t  = torch.tensor([cidx], dtype=torch.long, device=DEVICE)

                y_samples_norm = mc_decode_samples(model, x_enc_t, x_dec_t, cidx_t, MC_SAMPLES)  # [S,T]
                # inverse-transform efficiently
                y_den_flat = sy.inverse_transform(y_samples_norm.reshape(-1,1)).reshape(MC_SAMPLES, -1)  # [S,T]

                mu  = y_den_flat.mean(axis=0)
                std = y_den_flat.std(axis=0, ddof=0)
                qlo = np.quantile(y_den_flat, MC_Q_LO, axis=0)
                qhi = np.quantile(y_den_flat, MC_Q_HI, axis=0)

                out['co2_mu']   = mu
                out['co2_std']  = std
                out['co2_q%02d' % int(MC_Q_LO*100)] = qlo
                out['co2_q%02d' % int(MC_Q_HI*100)] = qhi

            scenario_results.append(out)

        if scenario_results:
            all_results.extend(scenario_results)

    if not all_results:
        return None, {"country": country_code, "reason": "no_forecast_generated"}, None

    df_res_full = pd.concat(all_results, ignore_index=True)

    # Save deterministic-only view (to preserve old pipeline compatibility)
    det_cols = ['year'] + VARIABLES + ['co2_pred', 'UID', 'GID_0', 'scenario']
    det_cols = [c for c in det_cols if c in df_res_full.columns]
    df_res_det = df_res_full[det_cols].copy()
    fp_city = os.path.join(OUT_DIR_CITY, f"{country_code.lower()}_cities_scen_v19.csv")
    df_res_det.to_csv(fp_city, index=False)

    # Save MC-augmented view if enabled
    if USE_MC:
        fp_city_mc = os.path.join(OUT_DIR_CITY, f"{country_code.lower()}_cities_scen_v19_stats.csv")
        df_res_full.to_csv(fp_city_mc, index=False)
    else:
        fp_city_mc = None

    # Summary (deterministic means)
    summary = {
        "country": country_code,
        "n_cities": int(df_res_det['UID'].nunique()),
        "train_r2": None if pd.isna(tr_r2) else float(tr_r2),
        "train_rmse": None if pd.isna(tr_rmse) else float(tr_rmse),
        "val_r2": None if pd.isna(va_r2) else float(va_r2),
        "val_rmse": None if pd.isna(va_rmse) else float(va_rmse),
    }
    for sc in SCENARIOS:
        sc_df = df_res_det[df_res_det['scenario'] == sc]
        if sc_df.empty:
            summary[f"{sc.lower()}_2021_mean"] = None
            summary[f"{sc.lower()}_2050_mean"] = None
            summary[f"{sc.lower()}_change_pct"] = None
            continue
        avg_2021 = sc_df[sc_df['year'] == 2021]['co2_pred'].mean()
        avg_2050 = sc_df[sc_df['year'] == 2050]['co2_pred'].mean()
        change_rate = ((avg_2050 - avg_2021) / avg_2021 * 100.0) if avg_2021 != 0 else np.nan
        summary[f"{sc.lower()}_2021_mean"] = float(avg_2021) if not pd.isna(avg_2021) else None
        summary[f"{sc.lower()}_2050_mean"] = float(avg_2050) if not pd.isna(avg_2050) else None
        summary[f"{sc.lower()}_change_pct"] = float(change_rate) if not pd.isna(change_rate) else None

    return df_res_det, df_res_full, summary, fp_city, fp_city_mc

if __name__ == "__main__":
    print("Loading datasets...")
    df = pd.read_csv(CITY_DATA_PATH).copy()

    ssp = pd.read_csv(SSP_DATA_PATH)
    ssp_pivot = {}
    for sc in [s for s in SCENARIOS if s != 'BAU']:
        if sc in ssp['Scenario'].values:
            df_sc = ssp[ssp['Scenario'] == sc].pivot(index='Year', columns='Variable', values='Value')
            full_years = np.arange(df_sc.index.min(), df_sc.index.max() + 1)
            df_sc = df_sc.reindex(full_years).interpolate().ffill().bfill()
            ssp_pivot[sc] = df_sc
            print(f"  Loaded {sc}: {len(df_sc)} years, {len(df_sc.columns)} variables")
        else:
            print(f"  Warning: {sc} not found in SSP data")

    all_forecasts_det, all_forecasts_mc, all_summaries, skipped = [], [], [], []

    for cc in COUNTRY_CODES:
        print(f"\n=== Processing {cc} ===")
        res_det, res_full, summary, fp_city, fp_city_mc = train_and_forecast_one_country(df, ssp_pivot, cc)
        if res_det is None:
            print(f"  Skipped {cc}: {summary.get('reason')}")
            skipped.append(summary); all_summaries.append(summary); continue
        all_forecasts_det.append(res_det)
        if res_full is not None:
            all_forecasts_mc.append(res_full)
        all_summaries.append(summary)
        print(f"    Saved deterministic: {fp_city}")
        if fp_city_mc:
            print(f"    Saved MC stats:     {fp_city_mc}")

    if all_forecasts_det:
        df_all_det = pd.concat(all_forecasts_det, ignore_index=True)
        fp_all_det = os.path.join(OUT_DIR_CITY, "annexI_cities_scen_v19.csv")
        df_all_det.to_csv(fp_all_det, index=False)
        print(f"\nSaved ALL-COUNTRY deterministic forecasts to {fp_all_det}")

    if USE_MC and all_forecasts_mc:
        df_all_mc = pd.concat(all_forecasts_mc, ignore_index=True)
        fp_all_mc = os.path.join(OUT_DIR_CITY, "annexI_cities_scen_v19_stats.csv")
        df_all_mc.to_csv(fp_all_mc, index=False)
        print(f"Saved ALL-COUNTRY MC stats to {fp_all_mc}")

    df_sum = pd.DataFrame(all_summaries)
    fp_summary = os.path.join(OUT_DIR_SUMMARY, "annexI_summary_v19.csv")
    df_sum.to_csv(fp_summary, index=False)
    print(f"Saved summary to {fp_summary}")

    fp_skipped = os.path.join(OUT_DIR_SUMMARY, "annexI_skipped_v19.json")
    with open(fp_skipped, "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)
    print(f"Saved skipped-country reasons to {fp_skipped}")

    print("\nMC Dropout Uncertainty enabled:" if USE_MC else "\nMC Dropout Uncertainty disabled.")
    print("Finished.")
