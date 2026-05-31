"""
Standalone UVR5 separation script.
Usage: python run_uvr5_sep.py
"""
import os
import sys

# Set required env vars before importing project modules
os.environ["weight_uvr5_root"] = "assets/uvr5_weights"
os.environ["weight_root"] = "assets/weights"
os.environ["index_root"] = "logs"
os.environ["outside_index_root"] = "assets/indices"
os.environ["rmvpe_root"] = "assets/rmvpe"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import soundfile as sf
import librosa

from infer.lib.uvr5_pack.lib_v5 import nets_61968KB as Nets
from infer.lib.uvr5_pack.lib_v5 import spec_utils
from infer.lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
from infer.lib.uvr5_pack.utils import inference

INPUT_FILE = r"E:\Data_voice\lonely.mp3"
OUT_VOCAL = r"E:\Data_voice\uvr5_out"
OUT_INS   = r"E:\Data_voice\uvr5_out"
MODEL_NAME = "HP5_only_main_vocal"
AGG = 10
FORMAT = "wav"

os.makedirs(OUT_VOCAL, exist_ok=True)
os.makedirs(OUT_INS, exist_ok=True)

# --------------- device ---------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    is_half = True
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    is_half = False
    print("Using CPU")

# --------------- load model ---------------
model_path = os.path.join("assets/uvr5_weights", MODEL_NAME + ".pth")
print(f"Loading model: {model_path}")
mp = ModelParameters("infer/lib/uvr5_pack/lib_v5/modelparams/4band_v2.json")
model = Nets.CascadedASPPNet(mp.param["bins"] * 2)
cpk = torch.load(model_path, map_location="cpu")
model.load_state_dict(cpk)
model.eval()
if is_half:
    model = model.half().to(device)
else:
    model = model.to(device)

# --------------- reformat input to stereo 44100 ---------------
import tempfile
tmp_path = os.path.join(tempfile.gettempdir(), "lonely_reformatted.wav")
cmd = f'ffmpeg -i "{INPUT_FILE}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
print(f"Reformatting: {cmd}")
ret = os.system(cmd)
print(f"Reformat exit code: {ret}")
assert os.path.exists(tmp_path), f"Reformat failed: {tmp_path}"

# --------------- process bands ---------------
data = {
    "postprocess": False,
    "tta": False,
    "window_size": 512,
    "agg": AGG,
    "high_end_process": "mirroring",
}

X_wave, X_spec_s = {}, {}
bands_n = len(mp.param["band"])
for d in range(bands_n, 0, -1):
    bp = mp.param["band"][d]
    if d == bands_n:
        X_wave[d], _ = librosa.load(
            tmp_path, sr=bp["sr"], mono=False, dtype=np.float32, res_type=bp["res_type"]
        )
        if X_wave[d].ndim == 1:
            X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
    else:
        X_wave[d] = librosa.resample(
            X_wave[d + 1],
            orig_sr=mp.param["band"][d + 1]["sr"],
            target_sr=bp["sr"],
            res_type=bp["res_type"],
        )
    X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
        X_wave[d], bp["hl"], bp["n_fft"],
        mp.param["mid_side"], mp.param["mid_side_b2"], mp.param["reverse"],
    )
    if d == bands_n and data["high_end_process"] != "none":
        input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
            mp.param["pre_filter_stop"] - mp.param["pre_filter_start"]
        )
        input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :]
        input_high_end = np.nan_to_num(np.array(input_high_end, dtype=np.complex128))

X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
aggresive_set = float(data["agg"] / 100)
aggressiveness = {"value": aggresive_set, "split_bin": mp.param["band"][1]["crop_stop"]}

print("Running neural inference...")
with torch.no_grad():
    pred, X_mag, X_phase = inference(X_spec_m, device, model, aggressiveness, data)

# pred may be float16 (half-precision) — cast to float32 to avoid NaN/Inf overflow
# X_phase is complex (np.exp(1j*angle)), X_spec_m is complex float64
pred = np.array(pred, dtype=np.float32)
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

# X_spec_m may also carry fp16 artifacts; ensure float64 complex
X_spec_m = np.array(X_spec_m, dtype=np.complex128)
X_spec_m = np.nan_to_num(X_spec_m, nan=0.0, posinf=0.0, neginf=0.0)

# X_phase is already complex from np.exp(1j*...)
X_phase = np.nan_to_num(X_phase, nan=0.0, posinf=0.0, neginf=0.0)

y_spec_m = pred * X_phase      # instrumental spectrogram
v_spec_m = X_spec_m - y_spec_m # vocal spectrogram

# --------------- save instrumental ---------------
input_high_end_ = spec_utils.mirroring(data["high_end_process"], y_spec_m, input_high_end, mp)
wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)
name = os.path.basename(tmp_path)
ins_path = os.path.join(OUT_INS, f"instrument_{name}_{AGG}.{FORMAT}")
sf.write(ins_path, (np.array(wav_instrument) * 32768).astype("int16"), mp.param["sr"])
print(f"Instrumental saved: {ins_path}")

# --------------- save vocal ---------------
input_high_end_ = spec_utils.mirroring(data["high_end_process"], v_spec_m, input_high_end, mp)
wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp, input_high_end_h, input_high_end_)
voc_path = os.path.join(OUT_VOCAL, f"vocal_{name}_{AGG}.{FORMAT}")
sf.write(voc_path, (np.array(wav_vocals) * 32768).astype("int16"), mp.param["sr"])
print(f"Vocal saved: {voc_path}")

print("\nUVR5 separation COMPLETE.")
print(f"  Vocal:        {voc_path}")
print(f"  Instrumental: {ins_path}")
