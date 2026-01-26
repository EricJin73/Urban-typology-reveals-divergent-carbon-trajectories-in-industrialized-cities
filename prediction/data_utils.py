"""
Data Processing Tools - Extrapolation, Sequence Building, Scaling
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def extrapolate_variables_bau(df, vars_list, hist_start, hist_end, future_start, future_end):
    """
    Linear extrapolation for BAU scenario (with non-negative constraint).
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


def extrapolate_with_ssp(df, vars_list, hist_start, hist_end, 
                         future_start, future_end, ssp_pivot, scenario):
    """
    SSP-based hybrid extrapolation (Linear + SSP ratio).
    """
    df_histwin = df[(df['year'] >= hist_start) & (df['year'] <= hist_end)]
    if df_histwin.empty or hist_end not in df['year'].values:
        return None
    
    hist_vals = df[df['year'] == hist_end][vars_list].iloc[0]
    
    if scenario not in ssp_pivot:
        return None
    
    ssp_df = ssp_pivot[scenario]
    
    # Extend SSP data range
    if future_start < ssp_df.index.min() or future_end > ssp_df.index.max():
        ssp_df = ssp_df.reindex(
            np.arange(min(ssp_df.index.min(), future_start),
                      max(ssp_df.index.max(), future_end) + 1)
        ).interpolate().ffill().bfill()
    
    ssp_future = ssp_df.loc[future_start:future_end]
    years_f = np.arange(future_start, future_end + 1)
    df_fut = pd.DataFrame({'year': years_f})
    
    df_hist = df_histwin.copy()
    df_hist['t'] = df_hist['year'] - hist_start
    
    for var in vars_list:
        # SSP ratio adjustment
        if var in ssp_df.columns and hist_end in ssp_df.index:
            base = ssp_df.at[hist_end, var]
            if base == 0:
                base = 1e-10
            ratio = ssp_future[var].values / base
            ssp_values = hist_vals[var] * ratio
        else:
            ssp_values = None
        
        # Linear extrapolation
        if df_hist[var].isna().all():
            linear_values = np.zeros_like(years_f, dtype=float)
        else:
            y = df_hist[var].values
            model = sm.OLS(y, sm.add_constant(df_hist['t'])).fit()
            linear_values = model.predict(sm.add_constant(years_f - hist_start))
        
        # Hybrid combination
        hybrid_values = linear_values if ssp_values is None else 0.5 * ssp_values + 0.5 * linear_values
        df_fut[var] = np.maximum(hybrid_values, 0)
    
    return df_fut


def build_city_sequences(df, city_idx, seq_len, pred_len, variables, hist_start, hist_end):
    """
    Construct encoder-decoder sequences for a single city.
    
    Returns:
        Xenc, Yenc, Xdec, Ydec, Y, idx (each is a numpy array)
    """
    df_h = df[(df['year'] >= hist_start) & (df['year'] <= hist_end)].sort_values('year')
    X = df_h[variables].values.astype(np.float32)
    y = df_h['co2'].values.astype(np.float32)
    years = df_h['year'].values
    
    X_enc, Y, X_dec, idx = [], [], [], []
    years_enc_list, years_dec_list = [], []
    
    for i in range(len(X) - seq_len - pred_len + 1):
        enc = X[i:i + seq_len]
        dec = X[i + seq_len: i + seq_len + pred_len]
        tgt = y[i + seq_len: i + seq_len + pred_len]
        yrs_enc = years[i:i + seq_len]
        yrs_dec = years[i + seq_len: i + seq_len + pred_len]
        
        X_enc.append(enc)
        X_dec.append(dec)
        Y.append(tgt)
        years_enc_list.append(yrs_enc)
        years_dec_list.append(yrs_dec)
        idx.append(city_idx)
    
    if not X_enc:
        return (np.zeros((0, seq_len, len(variables)), np.float32),
                np.zeros((0, seq_len), np.int32),
                np.zeros((0, pred_len, len(variables)), np.float32),
                np.zeros((0, pred_len), np.int32),
                np.zeros((0, pred_len), np.float32),
                np.zeros((0,), np.int64))
    
    return (np.array(X_enc), 
            np.array(years_enc_list, dtype=np.int32),
            np.array(X_dec), 
            np.array(years_dec_list, dtype=np.int32),
            np.array(Y), 
            np.array(idx, dtype=np.int64))


class CityScalerManager:
    """Manages feature and target scalers for each city."""
    
    def __init__(self):
        self.x_scalers = {}
        self.y_scalers = {}
    
    def fit(self, city, X_enc, X_dec, Y):
        """Fit scalers for a single city."""
        enc_flat = X_enc.reshape(-1, X_enc.shape[-1])
        dec_flat = X_dec.reshape(-1, X_dec.shape[-1])
        
        sx = StandardScaler().fit(np.vstack([enc_flat, dec_flat]))
        sy = StandardScaler().fit(Y.reshape(-1, 1))
        
        self.x_scalers[city] = sx
        self.y_scalers[city] = sy
    
    def transform_pair(self, city, X_enc, X_dec):
        """Transform encoder and decoder inputs."""
        sx = self.x_scalers[city]
        feat_dim = X_enc.shape[-1]
        
        enc_core = X_enc.reshape(-1, feat_dim)
        dec_core = X_dec.reshape(-1, feat_dim)
        
        enc_norm = sx.transform(enc_core).reshape(X_enc.shape).astype(np.float32)
        dec_norm = sx.transform(dec_core).reshape(X_dec.shape).astype(np.float32)
        
        return enc_norm, dec_norm
    
    def transform_y(self, city, Y):
        """Transform target values."""
        sy = self.y_scalers[city]
        return sy.transform(Y.reshape(-1, 1)).reshape(Y.shape)
    
    def inverse_transform_y(self, city, Y_norm):
        """Inverse transform target values."""
        sy = self.y_scalers[city]
        return sy.inverse_transform(Y_norm.reshape(-1, 1)).reshape(Y_norm.shape)
    
    def has_city(self, city):
        """Check if the city has scalers."""
        return city in self.x_scalers and city in self.y_scalers
    
    def get_scaler_list(self, city2idx):
        """Get list of y scalers sorted by index."""
        sy_list = [None] * len(city2idx)
        for city, idx in city2idx.items():
            if city in self.y_scalers:
                sy_list[idx] = self.y_scalers[city]
        return sy_list
