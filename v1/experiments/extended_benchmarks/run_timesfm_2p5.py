# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation script for timesfm 2.5."""

import os
import sys
import time
import multiprocessing

from absl import flags
import numpy as np
import pandas as pd
import timesfm
import torch

# Limit worker processes to prevent runaway multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

try:
  from .utils import ExperimentHandler
except Exception:
  try:
    from v1.experiments.extended_benchmarks.utils import ExperimentHandler
  except Exception:
    from utils import ExperimentHandler

dataset_names = [
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_other",
    "m3_quarterly",
    "m3_yearly",
    "m4_quarterly",
    "m4_yearly",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "nn5_daily_without_missing",
    "m5",
    "nn5_weekly",
    "traffic",
    "weather",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "ercot",
    "ett_small_15min",
    "ett_small_1h",
    "exchange_rate",
    "fred_md",
    "hospital",
]

context_dict_v1 = {
    "cif_2016": 32,
    "tourism_yearly": 64,
    "covid_deaths": 64,
    "tourism_quarterly": 64,
    "tourism_monthly": 64,
    "m1_monthly": 64,
    "m1_quarterly": 64,
    "m1_yearly": 64,
    "m3_monthly": 64,
    "m3_other": 64,
    "m3_quarterly": 64,
    "m3_yearly": 64,
    "m4_quarterly": 64,
    "m4_yearly": 64,
}

_MODEL_PATH = flags.DEFINE_string("model_path", "/scratch/wd04/sm0074/timesfm/models_pytorch/timesfm-2p5",
                                  "Path to model")
_HORIZON = flags.DEFINE_integer("horizon", 128, "Horizon")
_PER_CORE_BATCH_SIZE = flags.DEFINE_integer("per_core_batch_size", 8, "Per core batch size")
_SAVE_DIR = flags.DEFINE_string("save_dir", "./results", "Save directory")

QUANTILES = list(np.arange(1, 10) / 10.0)


def forecast_on_df_adapter(model, inputs, freq, value_name, forecast_context_len, horizon):
  """Adapter to convert dataframe input to TimesFM 2.5 format."""
  # Group by unique_id and convert to list of arrays
  grouped = inputs.groupby("unique_id")[value_name].apply(list).reset_index()
  
  time_series_list = []
  for idx, row in grouped.iterrows():
    ts = np.array(row[value_name], dtype=np.float32)
    # Take last forecast_context_len points if specified
    if forecast_context_len > 0 and len(ts) > forecast_context_len:
      ts = ts[-forecast_context_len:]
    time_series_list.append(ts)
  
  # Call the 2.5 forecast API
  point_forecasts, quantile_forecasts = model.forecast(
      horizon=horizon,
      inputs=time_series_list
  )
  
  # Convert back to dataframe format
  fcst_df_list = []
  for idx, row in grouped.iterrows():
    unique_id = row["unique_id"]
    # Get last timestamp from input
    last_time = inputs[inputs["unique_id"] == unique_id]["ds"].max()
    
    # Create future timestamps based on freq
    future_dates = pd.date_range(start=last_time, periods=horizon+1, freq=freq)[1:]
    
    fcst_data = {
        "unique_id": [unique_id] * horizon,
        "ds": future_dates,
        "timesfm": point_forecasts[idx, :],
    }
    
    # Add quantile columns
    for q_idx, q in enumerate(QUANTILES):
      fcst_data[f"timesfm-q-{q}"] = quantile_forecasts[idx, :, q_idx]
    
    fcst_df_list.append(pd.DataFrame(fcst_data))
  
  return pd.concat(fcst_df_list, ignore_index=True)


def main():
  os.environ['GLUONTS_DATASET_PATH'] = '/scratch/wd04/sm0074/timesfm/gluonts_cache'
  os.environ['HF_HUB_OFFLINE'] = '1'
  
  # Force single-threaded dataset loading to avoid worker explosion
  os.environ['GLUONTS_NUM_WORKERS'] = '0'
  os.environ['GLUONTS_POOL_WORKERS'] = '1'  # Critical: limits ExperimentHandler multiprocessing pool
  
  # Set multiprocessing start method to avoid fork issues
  try:
    multiprocessing.set_start_method('spawn', force=True)
  except RuntimeError:
    pass  # Already set
  
  print(f"Environment check:", flush=True)
  print(f"  GLUONTS_POOL_WORKERS={os.environ.get('GLUONTS_POOL_WORKERS', 'not set')}", flush=True)
  print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}", flush=True)
  
  results_list = []
  model_path = _MODEL_PATH.value
  horizon = _HORIZON.value
  
  # Load TimesFM 2.5 model
  print(f"Loading TimesFM 2.5 from {model_path}", flush=True)
  tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      model_path,
      local_files_only=True,
      torch_compile=False  # Disable torch.compile for compatibility
  )
  
  # Compile the model
  print("Compiling TimesFM 2.5 model...", flush=True)
  tfm.compile(
      timesfm.ForecastConfig(
          max_context=1024,
          max_horizon=horizon,
          per_core_batch_size=_PER_CORE_BATCH_SIZE.value,
          normalize_inputs=True,
          use_continuous_quantile_head=False,
          force_flip_invariance=True,
          infer_is_positive=True,
          fix_quantile_crossing=True,
      )
  )
  print("Model compiled successfully!", flush=True)
  
  run_id = np.random.randint(100000)
  model_name = "timesfm"
  skipped = []
  processed = 0
  
  for dataset in dataset_names:
    print(f"\n{'='*60}", flush=True)
    print(f"[{processed+1}/{len(dataset_names)}] Evaluating model {model_name} on dataset {dataset}", flush=True)
    print(f"{'='*60}", flush=True)
    
    try:
      print(f"Loading dataset {dataset}...", flush=True)
      exp = ExperimentHandler(dataset, quantiles=QUANTILES)
      print(f"Dataset {dataset} loaded successfully. Train shape: {exp.train_df.shape}", flush=True)
    except Exception as e:
      print(f"Skipping dataset {dataset}: failed to load experiment handler: {e}", flush=True)
      skipped.append(dataset)
      continue

    if dataset in context_dict_v1:
      context_len = context_dict_v1[dataset]
    else:
      context_len = 512

    try:
      train_df = exp.train_df
      freq = exp.freq
      num_series = train_df['unique_id'].nunique()
      print(f"Forecasting {num_series} time series with context_len={context_len}...", flush=True)
      
      init_time = time.time()
      
      # Use adapter function
      fcsts_df = forecast_on_df_adapter(
          model=tfm,
          inputs=train_df,
          freq=freq,
          value_name="y",
          forecast_context_len=context_len,
          horizon=horizon
      )
      
      total_time = time.time() - init_time
      print(f"Forecasting completed in {total_time:.2f}s ({total_time/num_series:.3f}s per series)", flush=True)
      
      print(f"Evaluating metrics...", flush=True)
      time_df = pd.DataFrame({"time": [total_time], "model": model_name})
      results = exp.evaluate_from_predictions(
          models=[model_name],
          fcsts_df=fcsts_df,
          times_df=time_df
      )
      print(f"Results for {dataset}:", flush=True)
      print(results, flush=True)
      results_list.append(results)
      processed += 1
      print(f"✓ Dataset {dataset} completed successfully!", flush=True)
    except Exception as e:
      import traceback
      print(f"Failed during evaluation for dataset {dataset}: {e}", flush=True)
      traceback.print_exc()
      skipped.append(dataset)
      continue

  # If nothing processed, exit gracefully
  if processed == 0:
    print("No datasets were successfully processed. Exiting gracefully.", flush=True)
    print(f"Skipped datasets: {skipped}", flush=True)
    return

  results_full = pd.concat(results_list)
  timestamp = time.strftime("%Y%m%dT%H%M%S")
  save_path = os.path.join(_SAVE_DIR.value, str(run_id))
  print(f"Saving results to {save_path}", flush=True)
  os.makedirs(save_path, exist_ok=True)
  results_full.to_csv(f"{save_path}/results_{timestamp}.csv")
  print(f"Successfully processed {processed} datasets. Skipped {len(skipped)} datasets.", flush=True)
  if skipped:
    print(f"Skipped datasets: {skipped}", flush=True)


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  main()
