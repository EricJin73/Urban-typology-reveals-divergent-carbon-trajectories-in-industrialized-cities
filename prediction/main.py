"""
Main Program - Run Forecasts for All Countries
"""
import json
import numpy as np
import pandas as pd

from config import Config
from pipeline import CountryPipeline


def load_ssp_data(config):
    """Load and process SSP scenario data."""
    ssp = pd.read_csv(config.SSP_DATA_PATH)
    ssp_pivot = {}
    
    for sc in [s for s in config.SCENARIOS if s != 'BAU']:
        if sc in ssp['Scenario'].values:
            df_sc = ssp[ssp['Scenario'] == sc].pivot(
                index='Year', 
                columns='Variable', 
                values='Value'
            )
            
            # Interpolation
            full_years = np.arange(df_sc.index.min(), df_sc.index.max() + 1)
            df_sc = df_sc.reindex(full_years).interpolate().ffill().bfill()
            
            ssp_pivot[sc] = df_sc
            print(f"  Loaded {sc}: {len(df_sc)} years, {len(df_sc.columns)} variables")
        else:
            print(f"  Warning: {sc} not found in SSP data")
    
    return ssp_pivot


def main():
    """Main function."""
    # Initialize configuration
    Config.init_dirs()
    Config.set_seed()
    
    print("=" * 60)
    print("FiLM Seq2Seq with Monte Carlo Dropout Uncertainty")
    print("=" * 60)
    
    # Load data
    print("\nLoading datasets...")
    df = pd.read_csv(Config.CITY_DATA_PATH).copy()
    print(f"  City data: {len(df)} rows")
    
    ssp_pivot = load_ssp_data(Config)
    
    # Initialize pipeline
    pipeline = CountryPipeline(Config)
    
    # Collect results
    all_forecasts_det = []
    all_forecasts_mc = []
    all_summaries = []
    skipped = []
    
    # Process each country
    for cc in Config.COUNTRY_CODES:
        print(f"\n{'='*60}")
        print(f"Processing {cc}")
        print('='*60)
        
        df_det, df_full, summary, fp_city, fp_city_mc = pipeline.run(df, ssp_pivot, cc)
        
        if df_det is None:
            print(f"  ⚠ Skipped {cc}: {summary.get('reason')}")
            skipped.append(summary)
            all_summaries.append(summary)
            continue
        
        all_forecasts_det.append(df_det)
        if df_full is not None:
            all_forecasts_mc.append(df_full)
        
        all_summaries.append(summary)
        
        print(f"  ✓ Saved deterministic: {fp_city}")
        if fp_city_mc:
            print(f"  ✓ Saved MC stats: {fp_city_mc}")
    
    # Merge results from all countries
    print(f"\n{'='*60}")
    print("Saving combined results...")
    print('='*60)
    
    if all_forecasts_det:
        df_all_det = pd.concat(all_forecasts_det, ignore_index=True)
        fp_all_det = f"{Config.OUT_DIR_CITY}/annexI_cities_scen_v19.csv"
        df_all_det.to_csv(fp_all_det, index=False)
        print(f"  ✓ All-country deterministic: {fp_all_det}")
    
    if Config.USE_MC and all_forecasts_mc:
        df_all_mc = pd.concat(all_forecasts_mc, ignore_index=True)
        fp_all_mc = f"{Config.OUT_DIR_CITY}/annexI_cities_scen_v19_stats.csv"
        df_all_mc.to_csv(fp_all_mc, index=False)
        print(f"  ✓ All-country MC stats: {fp_all_mc}")
    
    # Save summary
    df_sum = pd.DataFrame(all_summaries)
    fp_summary = f"{Config.OUT_DIR_SUMMARY}/annexI_summary_v19.csv"
    df_sum.to_csv(fp_summary, index=False)
    print(f"  ✓ Summary: {fp_summary}")
    
    # Save skipped country information
    fp_skipped = f"{Config.OUT_DIR_SUMMARY}/annexI_skipped_v19.json"
    with open(fp_skipped, "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Skipped countries: {fp_skipped}")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"  Total countries processed: {len(Config.COUNTRY_CODES)}")
    print(f"  Successfully completed: {len(all_forecasts_det)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  MC Dropout enabled: {Config.USE_MC}")
    print("\n✓ Finished!")


if __name__ == "__main__":
    main()
