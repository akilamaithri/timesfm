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
"""Evaluation script for timesfm."""

import os
import sys
import time

from absl import flags
import numpy as np
import pandas as pd
import timesfm
import torch
try:
  from safetensors import torch as safetorch  # type: ignore
except Exception:
  safetorch = None

try:
  # When executed as a module within the package
  from .utils import ExperimentHandler
except Exception:
  try:
    # When imported as a top-level package module
    from v1.experiments.extended_benchmarks.utils import ExperimentHandler
  except Exception:
    # When executed directly as a script from the working dir
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


context_dict_v2 = {}

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

_MODEL_PATH = flags.DEFINE_string("model_path", "google/timesfm-2.5-200m-pytorch",
                                  "Path to model")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Batch size")
_HORIZON = flags.DEFINE_integer("horizon", 128, "Horizon")
_BACKEND = flags.DEFINE_string("backend", "gpu", "Backend")
_NUM_JOBS = flags.DEFINE_integer("num_jobs", 1, "Number of jobs")
_SAVE_DIR = flags.DEFINE_string("save_dir", "./results", "Save directory")

QUANTILES = list(np.arange(1, 10) / 10.0)


def main():
  os.environ['GLUONTS_DATASET_PATH'] = '/scratch/wd04/sm0074/timesfm/gluonts_cache'
  os.environ['HF_HUB_OFFLINE'] = '1'
  results_list = []
  model_path = _MODEL_PATH.value
  num_layers = 20
  max_context_len = 512
  use_positional_embedding = True
  context_dict = context_dict_v1

  # Default local checkpoint paths (already downloaded to the node).
  local_ckpt_2p5 = "/scratch/wd04/sm0074/timesfm/models_pytorch/model.safetensors"
  local_ckpt_2p0 = "/scratch/wd04/sm0074/timesfm/models_pytorch/torch_model.ckpt"
  # Local full snapshot directory (contains config.json + model.safetensors)
  local_snapshot_2p5 = "/scratch/wd04/sm0074/timesfm/models_pytorch/timesfm-2p5"

  tfm = None
  # Prefer 2.5 model if requested
  if "2.5" in model_path:
    # 2.5 is a smaller PyTorch model; set smaller defaults
    num_layers = 20
    max_context_len = 1024
    context_dict = context_dict_v1
    # Try to load using the 2.5 Torch helper if available; fall back to generic loader
    # If the local checkpoint is a safetensors file, attempt to convert it to
    # a PyTorch checkpoint on-disk so existing loaders (which call torch.load)
    # can succeed in an offline environment.
    ckpt_path = None
    if os.path.exists(local_ckpt_2p5):
      ckpt_path = local_ckpt_2p5
    # If safetensors file present and safetensors installed, convert it
    if ckpt_path and ckpt_path.endswith(".safetensors"):
      if safetorch is None:
        print("safetensors not installed; cannot load .safetensors checkpoint.\n"
              "Install safetensors in the environment or provide a PyTorch checkpoint.", flush=True)
      else:
        try:
          print(f"Converting safetensors checkpoint {ckpt_path} -> torch .pt temporary file", flush=True)
          state = safetorch.load_file(ckpt_path, device="cpu")
          converted = f"{ckpt_path}.pt"
          torch.save(state, converted)
          ckpt_path = converted
          print(f"Converted checkpoint saved to {converted}", flush=True)
        except Exception as e:
          print(f"Failed to convert safetensors checkpoint: {e}", flush=True)

    # Prefer loading from a local full snapshot if present (contains config + model)
    try:
      if os.path.isdir(local_snapshot_2p5):
        # Attempt to load using the local snapshot directory so config and weights match.
        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
          # If we converted a safetensors file to a .pt earlier, prefer that
          # explicit checkpoint_path so the loader doesn't attempt to torch.load
          # the original safetensors file.
          if ckpt_path and ckpt_path.endswith('.pt'):
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(local_snapshot_2p5, local_files_only=True, checkpoint_path=ckpt_path)
          else:
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(local_snapshot_2p5, local_files_only=True)
        else:
          # Fallback to generic loader: point to the actual model file inside
          # the snapshot (do not pass the directory itself as a file path).
          snapshot_model_file = None
          for candidate in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt", "torch_model.ckpt"):
            cand_path = os.path.join(local_snapshot_2p5, candidate)
            if os.path.exists(cand_path):
              snapshot_model_file = cand_path
              break
          if snapshot_model_file is None:
            raise FileNotFoundError(f"No model file found inside snapshot {local_snapshot_2p5}")
          # If we converted safetensors to a .pt earlier, prefer that file when
          # instantiating the generic loader so it doesn't try to torch.load the
          # safetensors directly.
          load_file = ckpt_path if (ckpt_path and ckpt_path.endswith('.pt')) else snapshot_model_file
          tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
              backend=_BACKEND.value,
              per_core_batch_size=32,
              horizon_len=_HORIZON.value,
              num_layers=num_layers,
              context_len=max_context_len,
              use_positional_embedding=use_positional_embedding,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path, path=load_file),
          )
      else:
        # Existing behavior: try using a single checkpoint file (converted .pt or .safetensors)
        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
          if ckpt_path:
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path, local_files_only=True, checkpoint_path=ckpt_path)
          else:
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path, local_files_only=True)
        else:
          raise AttributeError("TimesFM_2p5_200M_torch loader not present in timesfm package")
    except Exception as e:
      print(f"Failed to load 2.5 model offline: {e}", flush=True)
      return
  elif "2.0" in model_path:
    # settings for 2.0
    num_layers = 50
    use_positional_embedding = False
    max_context_len = 2048
    context_dict = context_dict_v2
    # If running offline, provide a direct local checkpoint path so the loader
    # doesn't attempt Hub lookups. Adjust this path to the downloaded checkpoint
    # file present under `/scratch/wd04/sm0074/timesfm/models`.
    try:
      tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
          backend=_BACKEND.value,
          per_core_batch_size=32,
          horizon_len=_HORIZON.value,
          num_layers=num_layers,
          context_len=max_context_len,
          use_positional_embedding=use_positional_embedding,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path, path=local_ckpt_2p0),
      )
    except Exception as e:
      print(f"Failed to load 2.0 model offline: {e}", flush=True)
      return
  else:
    # Generic attempt for other model strings
    try:
      tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
          backend=_BACKEND.value,
          per_core_batch_size=32,
          horizon_len=_HORIZON.value,
          num_layers=num_layers,
          context_len=max_context_len,
          use_positional_embedding=use_positional_embedding,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
      )
    except Exception as e:
      print(f"Failed to load model {model_path}: {e}", flush=True)
      return
  run_id = np.random.randint(100000)
  model_name = "timesfm"
  skipped = []
  processed = 0
  for dataset in dataset_names:
    print(f"Evaluating model {model_name} on dataset {dataset}", flush=True)
    try:
      exp = ExperimentHandler(dataset, quantiles=QUANTILES)
    except Exception as e:
      print(f"Skipping dataset {dataset}: failed to load experiment handler: {e}", flush=True)
      skipped.append(dataset)
      continue

    if dataset in context_dict:
      context_len = context_dict[dataset]
    else:
      context_len = max_context_len

    try:
      train_df = exp.train_df
      freq = exp.freq
      init_time = time.time()
      fcsts_df = tfm.forecast_on_df(
          inputs=train_df,
          freq=freq,
          value_name="y",
          model_name=model_name,
          forecast_context_len=context_len,
          num_jobs=_NUM_JOBS.value,
          normalize=True,
      )
      total_time = time.time() - init_time
      time_df = pd.DataFrame({"time": [total_time], "model": model_name})
      results = exp.evaluate_from_predictions(models=[model_name],
                                              fcsts_df=fcsts_df,
                                              times_df=time_df)
      print(results, flush=True)
      results_list.append(results)
      processed += 1
    except Exception as e:
      print(f"Failed during evaluation for dataset {dataset}: {e}", flush=True)
      skipped.append(dataset)
      continue

  # If nothing processed, exit gracefully
  if processed == 0:
    print("No datasets were successfully processed. Exiting gracefully.", flush=True)
    return

  results_full = pd.concat(results_list)
  # Timestamped filename so multiple runs don't clobber each other
  timestamp = time.strftime("%Y%m%dT%H%M%S")
  save_path = os.path.join(_SAVE_DIR.value, str(run_id))
  print(f"Saving results to {save_path}", flush=True)
  os.makedirs(save_path, exist_ok=True)
  results_full.to_csv(f"{save_path}/results_{timestamp}.csv")


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  main()
