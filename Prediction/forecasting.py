"""
Scenario Forecasting and Uncertainty Estimation
"""
import numpy as np
import pandas as pd
import torch
from data_utils import extrapolate_variables_bau, extrapolate_with_ssp
from training import mc_dropout_predict


class ScenarioForecaster:
    """Scenario Forecaster"""
    
    def __init__(self, model, scaler_manager, city2idx, config):
        self.model = model
        self.scaler_manager = scaler_manager
        self.city2idx = city2idx
        self.idx2city = {i: c for c, i in city2idx.items()}
        self.config = config
    
    def forecast_all_scenarios(self, df_single, ssp_pivot, country_code):
        """Generate forecasts for all scenarios and cities."""
        all_results = []
        
        for scenario in self.config.SCENARIOS:
            scenario_results = self._forecast_scenario(
                df_single, ssp_pivot, scenario, country_code
            )
            if scenario_results:
                all_results.extend(scenario_results)
        
        if not all_results:
            return None, None
        
        df_full = pd.concat(all_results, ignore_index=True)
        
        # Deterministic results
        det_cols = ['year'] + self.config.VARIABLES + ['co2_pred', 'UID', 'GID_0', 'scenario']
        det_cols = [c for c in det_cols if c in df_full.columns]
        df_det = df_full[det_cols].copy()
        
        return df_det, df_full
    
    def _forecast_scenario(self, df_single, ssp_pivot, scenario, country_code):
        """Forecast all cities for a single scenario."""
        results = []
        
        for city, cidx in self.city2idx.items():
            result = self._forecast_city(
                df_single, city, cidx, scenario, ssp_pivot, country_code
            )
            if result is not None:
                results.append(result)
        
        return results
    
    def _forecast_city(self, df_single, city, cidx, scenario, ssp_pivot, country_code):
        """Generate forecasts for a single city and scenario."""
        df_c = df_single[df_single['UID'] == city].sort_values('year')
        
        # Extrapolate features
        if scenario == 'BAU':
            df_fut = extrapolate_variables_bau(
                df_c, self.config.VARIABLES,
                self.config.HIST_START, self.config.HIST_END,
                self.config.FUTURE_START, self.config.FUTURE_END
            )
        else:
            df_fut = extrapolate_with_ssp(
                df_c, self.config.VARIABLES,
                self.config.HIST_START, self.config.HIST_END,
                self.config.FUTURE_START, self.config.FUTURE_END,
                ssp_pivot, scenario
            )
        
        if df_fut is None:
            return None
        
        # Prepare encoder input (historical window)
        hist_feats = df_c[df_c['year'] <= self.config.HIST_END][self.config.VARIABLES].values.astype(np.float32)
        if len(hist_feats) < self.config.SEQ_LEN:
            return None
        
        enc_window = hist_feats[-self.config.SEQ_LEN:]
        enc_input = enc_window[None, ...]  # [1, L, F]
        
        # Prepare decoder input (future features)
        dec_feats = df_fut[self.config.VARIABLES].values.astype(np.float32)
        dec_input = dec_feats[None, ...]  # [1, T, F]
        
        # Scaling
        if not self.scaler_manager.has_city(city):
            return None
        
        x_enc, x_dec = self._scale_inputs(enc_input, dec_input, city)
        
        # Deterministic prediction
        y_norm = self._predict_deterministic(x_enc, x_dec, cidx)
        y_den = self.scaler_manager.inverse_transform_y(city, y_norm)
        
        # Build output
        out = df_fut.copy()
        out['co2_pred'] = y_den
        out['UID'] = city
        out['GID_0'] = country_code
        out['scenario'] = scenario
        
        # Monte Carlo uncertainty estimation
        if self.config.USE_MC:
            self._add_mc_uncertainty(out, x_enc, x_dec, cidx, city)
        
        return out
    
    def _scale_inputs(self, enc_input, dec_input, city):
        """Scale encoder and decoder inputs."""
        sx = self.scaler_manager.x_scalers[city]
        feat_dim = len(self.config.VARIABLES)
        
        enc_core = enc_input[..., :feat_dim].reshape(-1, feat_dim)
        enc_norm = sx.transform(enc_core).reshape(1, self.config.SEQ_LEN, feat_dim).astype(np.float32)
        
        dec_core = dec_input[..., :feat_dim].reshape(-1, feat_dim)
        dec_norm = sx.transform(dec_core).reshape(1, dec_input.shape[1], feat_dim).astype(np.float32)
        
        return enc_norm, dec_norm
    
    def _predict_deterministic(self, x_enc, x_dec, cidx):
        """Deterministic prediction."""
        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(
                torch.from_numpy(x_enc).to(self.config.DEVICE),
                torch.tensor([cidx], dtype=torch.long, device=self.config.DEVICE),
                torch.from_numpy(x_dec).to(self.config.DEVICE)
            ).cpu().numpy().reshape(-1)
        return y_norm
    
    def _add_mc_uncertainty(self, out, x_enc, x_dec, cidx, city):
        """Add Monte Carlo uncertainty statistics."""
        x_enc_t = torch.from_numpy(x_enc).float().to(self.config.DEVICE)
        x_dec_t = torch.from_numpy(x_dec).float().to(self.config.DEVICE)
        cidx_t = torch.tensor([cidx], dtype=torch.long, device=self.config.DEVICE)
        
        # MC sampling
        y_samples_norm = mc_dropout_predict(
            self.model, x_enc_t, x_dec_t, cidx_t, self.config.MC_SAMPLES
        )
        
        # Inverse transform
        sy = self.scaler_manager.y_scalers[city]
        y_samples = sy.inverse_transform(y_samples_norm.reshape(-1, 1)).reshape(
            self.config.MC_SAMPLES, -1
        )
        
        # Statistics
        out['co2_mu'] = y_samples.mean(axis=0)
        out['co2_std'] = y_samples.std(axis=0, ddof=0)
        out[f'co2_q{int(self.config.MC_Q_LO*100):02d}'] = np.quantile(
            y_samples, self.config.MC_Q_LO, axis=0
        )
        out[f'co2_q{int(self.config.MC_Q_HI*100):02d}'] = np.quantile(
            y_samples, self.config.MC_Q_HI, axis=0
        )
