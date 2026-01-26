# FiLM-Seq2Seq

## Urban typology reveals divergent carbon trajectories in industrialized cities

**Abstract**: Urban areas are central to global climate mitigation, yet existing projections often rely on coarse-grained assessment models that overlook the interplay between urban typology and socioeconomic dynamics. Here we harness a context-aware deep learning framework combined with typological clustering to project CO₂ emissions for 46,833 cities in Annex I countries through 2050 under Shared Socioeconomic Pathways. We identify five distinct urban typologies characterized by differing patterns of horizontal expansion, economic intensification, and vertical growth. We find that emission trajectories diverge significantly across these typologies: 'Vertically densifying' cities are projected to contribute 34.5% of total emission reductions under the SSP1. Conversely, 'Compact intensifying' cities exhibit a rebound in emissions post-2035. Our results suggest that uniform national policies are unsuitable for diverse urban typologies, necessitating type-specific strategies to achieve the full decarbonization potential of cities.

## Requirements

Plaintext
python==3.10.18
numpy==1.26.4
pandas==2.3.1
scikit-learn==1.7.1
scipy==1.15.3
statsmodels==0.14.5
torch==2.5.1

## Prediction

`config.py`: Centralized management for all hyperparameters, paths, and constants. Provides methods for directory initialization and random seed setup to ensure reproducibility.

`data_utils.py`: Handles data preprocessing including BAU (linear) and SSP (mixed) scenario extrapolation, sequence building, and unified scaling via CityScalerManager.

`model.py`: Defines the FiLMSeq2Seq architecture with a clear Encoder-Decoder structure and built-in Luong Attention mechanism for sequence mapping.

`training.py`: Encapsulates the training process within ModelTrainer. Supports R²/RMSE calculation and mc_dropout_predict for uncertainty estimation.

`forecasting.py`: Managed by ScenarioForecaster to automate feature extrapolation and model inference with MC sampling for statistical uncertainty results.

`pipeline.py`: Orchestrates the end-to-end CountryPipeline, streamlining the workflow from data preparation and training to validation and final saving.

`main.py`: The primary entry point. Run this script to execute the complete forecasting pipeline.

**Usage**
To start the full pipeline, run the following command:

```bash
python main.py
