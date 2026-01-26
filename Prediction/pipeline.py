"""
End-to-End Training and Forecasting Pipeline
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_utils import build_city_sequences, CityScalerManager
from model import FiLMSeq2Seq
from training import ModelTrainer, calculate_metrics, predict_batch
from forecasting import ScenarioForecaster


class CountryPipeline:
    """Complete training and forecasting pipeline for a single country."""
    
    def __init__(self, config):
        self.config = config
    
    def run(self, df_all, ssp_pivot, country_code):
        """
        Execute the complete pipeline.
        
        Returns:
            df_det, df_full, summary, fp_city, fp_city_mc
        """
        # 1. Data Preparation
        data = self._prepare_data(df_all, country_code)
        if data is None:
            return None, None, {"country": country_code, "reason": "data_preparation_failed"}, None, None
        
        city2idx, per_city, scaler_manager = data
        
        # 2. Build Training/Validation Sets
        train_val_data = self._build_train_val_data(per_city, city2idx, scaler_manager)
        if train_val_data is None:
            return None, None, {"country": country_code, "reason": "train_val_build_failed"}, None, None
        
        train_loader, val_loader, sy_list = train_val_data
        
        # 3. Train Model
        model, trainer = self._train_model(train_loader, val_loader, city2idx, country_code)
        
        # 4. Save Training History
        self._save_training_history(trainer, country_code)
        
        # 5. Calculate Metrics
        metrics = self._compute_metrics(model, train_loader, val_loader, sy_list)
        
        # 6. Export Train/Val Predictions
        self._export_train_val_predictions(
            model, per_city, city2idx, scaler_manager, country_code
        )
        
        # 7. Scenario Forecasting
        df_single = df_all[df_all['GID_0'] == country_code]
        forecaster = ScenarioForecaster(model, scaler_manager, city2idx, self.config)
        df_det, df_full = forecaster.forecast_all_scenarios(df_single, ssp_pivot, country_code)
        
        if df_det is None:
            return None, None, {"country": country_code, "reason": "no_forecast_generated"}, None, None
        
        # 8. Save Results
        fp_city, fp_city_mc = self._save_forecasts(df_det, df_full, country_code)
        
        # 9. Generate Summary
        summary = self._create_summary(country_code, df_det, metrics)
        
        return df_det, df_full, summary, fp_city, fp_city_mc
    
    def _prepare_data(self, df_all, country_code):
        """Prepare data for a single country."""
        df_single = df_all[df_all['GID_0'] == country_code]
        if df_single.empty:
            return None
        
        cities = df_single['UID'].unique()
        if len(cities) == 0:
            return None
        
        city2idx = {city: i for i, city in enumerate(cities)}
        per_city = {}
        scaler_manager = CityScalerManager()
        
        for city in cities:
            df_c = df_single[df_single['UID'] == city].sort_values('year')
            
            # Check data integrity
            if (df_c['year'].nunique() < self.config.SEQ_LEN + self.config.PRED_LEN_TRAIN or
                df_c[['co2'] + self.config.VARIABLES].isna().any().any()):
                continue
            
            # Build sequences
            Xenc, Yenc, Xdec, Ydec, Y, idx = build_city_sequences(
                df_c, city2idx[city], self.config.SEQ_LEN,
                self.config.PRED_LEN_TRAIN, self.config.VARIABLES,
                self.config.HIST_START, self.config.HIST_END
            )
            
            if len(Xenc) == 0:
                continue
            
            # Train/Validation Split
            n_samples = len(Xenc)
            train_size = max(1, int(0.8 * n_samples))
            perm = np.random.permutation(n_samples)
            tr, va = perm[:train_size], perm[train_size:]
            
            per_city[city] = {
                "Xenc_tr": Xenc[tr], "Yenc_tr": Yenc[tr],
                "Xdec_tr": Xdec[tr], "Ydec_tr": Ydec[tr], "Y_tr": Y[tr],
                "Xenc_va": Xenc[va], "Yenc_va": Yenc[va],
                "Xdec_va": Xdec[va], "Ydec_va": Ydec[va], "Y_va": Y[va]
            }
            
            # Fit scalers
            scaler_manager.fit(city, Xenc[tr], Xdec[tr], Y[tr])
        
        if not per_city:
            return None
        
        return city2idx, per_city, scaler_manager
    
    def _build_train_val_data(self, per_city, city2idx, scaler_manager):
        """Build training and validation data loaders."""
        Xenc_tr_list, Xdec_tr_list, y_tr_list = [], [], []
        city_idx_tr_list, years_tr_list = [], []
        Xenc_va_list, Xdec_va_list, y_va_list = [], [], []
        city_idx_va_list, years_va_list = [], []
        
        for city, d in per_city.items():
            if not scaler_manager.has_city(city):
                continue
            
            # Scale training data
            Xenc_tr, Xdec_tr = scaler_manager.transform_pair(city, d["Xenc_tr"], d["Xdec_tr"])
            y_tr = scaler_manager.transform_y(city, d["Y_tr"])
            
            # Scale validation data
            Xenc_va, Xdec_va = scaler_manager.transform_pair(city, d["Xenc_va"], d["Xdec_va"])
            y_va = scaler_manager.transform_y(city, d["Y_va"])
            
            # Training set
            Xenc_tr_list.append(Xenc_tr)
            Xdec_tr_list.append(Xdec_tr)
            y_tr_list.append(y_tr)
            city_idx_tr_list.append(np.full(len(Xenc_tr), city2idx[city], dtype=np.int64))
            years_tr_list.append(d["Ydec_tr"])
            
            # Validation set
            Xenc_va_list.append(Xenc_va)
            Xdec_va_list.append(Xdec_va)
            y_va_list.append(y_va)
            city_idx_va_list.append(np.full(len(Xenc_va), city2idx[city], dtype=np.int64))
            years_va_list.append(d["Ydec_va"])
        
        if not Xenc_tr_list:
            return None
        
        # Concatenate all cities
        Xenc_train = np.concatenate(Xenc_tr_list, axis=0)
        Xdec_train = np.concatenate(Xdec_tr_list, axis=0)
        y_train = np.concatenate(y_tr_list, axis=0)
        city_idx_train = np.concatenate(city_idx_tr_list, axis=0)
        years_train = np.concatenate(years_tr_list, axis=0)
        
        if Xenc_va_list:
            Xenc_val = np.concatenate(Xenc_va_list, axis=0)
            Xdec_val = np.concatenate(Xdec_va_list, axis=0)
            y_val = np.concatenate(y_va_list, axis=0)
            city_idx_val = np.concatenate(city_idx_va_list, axis=0)
            years_val = np.concatenate(years_va_list, axis=0)
        else:
            # Fallback: Split from training set
            n = len(Xenc_train)
            val_size = max(1, n // 10)
            Xenc_val, Xenc_train = Xenc_train[-val_size:], Xenc_train[:-val_size]
            Xdec_val, Xdec_train = Xdec_train[-val_size:], Xdec_train[:-val_size]
            y_val, y_train = y_train[-val_size:], y_train[:-val_size]
            city_idx_val, city_idx_train = city_idx_train[-val_size:], city_idx_train[:-val_size]
            years_val, years_train = years_train[-val_size:], years_train[:-val_size]
        
        # Create DataLoaders
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
        
        train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.BATCH_SIZE)
        
        sy_list = scaler_manager.get_scaler_list(city2idx)
        
        return train_loader, val_loader, sy_list
    
    def _train_model(self, train_loader, val_loader, city2idx, country_code):
        """Train the model."""
        model = FiLMSeq2Seq(
            n_cities=len(city2idx),
            emb_dim=self.config.EMB_DIM,
            feat_dim=len(self.config.VARIABLES),
            enc_hidden=self.config.ENC_HIDDEN,
            dec_hidden=self.config.DEC_HIDDEN,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT
        ).to(self.config.DEVICE)
        
        trainer = ModelTrainer(
            model, self.config.DEVICE,
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY,
            clip_norm=self.config.CLIP_NORM
        )
        
        trainer.fit(train_loader, val_loader, self.config.EPOCHS, 
                    verbose=True, country_code=country_code)
        
        return model, trainer
    
    def _save_training_history(self, trainer, country_code):
        """Save training history."""
        train_hist, val_hist = trainer.get_histories()
        
        pd.DataFrame({
            "epoch": np.arange(1, len(train_hist) + 1),
            "mse": train_hist
        }).to_csv(
            os.path.join(self.config.OUT_DIR_LOSS, 
                         f"{country_code.lower()}_train_loss_v19.csv"),
            index=False
        )
        
        pd.DataFrame({
            "epoch": np.arange(1, len(val_hist) + 1),
            "mse": val_hist
        }).to_csv(
            os.path.join(self.config.OUT_DIR_LOSS,
                         f"{country_code.lower()}_val_loss_v19.csv"),
            index=False
        )
    
    def _compute_metrics(self, model, train_loader, val_loader, sy_list):
        """Compute training and validation metrics."""
        tr_r2, tr_rmse = calculate_metrics(model, train_loader, self.config.DEVICE, sy_list)
        va_r2, va_rmse = calculate_metrics(model, val_loader, self.config.DEVICE, sy_list)
        
        return {
            "train_r2": tr_r2,
            "train_rmse": tr_rmse,
            "val_r2": va_r2,
            "val_rmse": va_rmse
        }
    
    def _export_train_val_predictions(self, model, per_city, city2idx, scaler_manager, country_code):
        """Export predictions for training and validation sets."""
        # This function is optional, simplified implementation here
        pass
    
    def _save_forecasts(self, df_det, df_full, country_code):
        """Save forecast results."""
        fp_city = os.path.join(
            self.config.OUT_DIR_CITY,
            f"{country_code.lower()}_cities_scen_v19.csv"
        )
        df_det.to_csv(fp_city, index=False)
        
        fp_city_mc = None
        if self.config.USE_MC and df_full is not None:
            fp_city_mc = os.path.join(
                self.config.OUT_DIR_CITY,
                f"{country_code.lower()}_cities_scen_v19_stats.csv"
            )
            df_full.to_csv(fp_city_mc, index=False)
        
        return fp_city, fp_city_mc
    
    def _create_summary(self, country_code, df_det, metrics):
        """Create summary statistics."""
        summary = {
            "country": country_code,
            "n_cities": int(df_det['UID'].nunique()),
            "train_r2": None if pd.isna(metrics["train_r2"]) else float(metrics["train_r2"]),
            "train_rmse": None if pd.isna(metrics["train_rmse"]) else float(metrics["train_rmse"]),
            "val_r2": None if pd.isna(metrics["val_r2"]) else float(metrics["val_r2"]),
            "val_rmse": None if pd.isna(metrics["val_rmse"]) else float(metrics["val_rmse"]),
        }
        
        for sc in self.config.SCENARIOS:
            sc_df = df_det[df_det['scenario'] == sc]
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
        
        return summary
