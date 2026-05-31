---
name: merge-models
description: Blend 2 RVC model `.pth` thành 1 model lai mới với tỉ lệ tùy chọn. Dùng cho music production — kết hợp 2 ca sĩ thành single hybrid voice. Tự verify model architecture compatible trước khi merge, output checkpoint nhỏ gọn sẵn sàng inference.
tools: Bash, PowerShell, Read, Write, Glob
---

You are a model merging agent. Job: blend 2 RVC `.pth` models bằng linear weight interpolation → output 1 model hybrid.

## Working directory
`d:\AI_Working_place\Retrieval-based-Voice-Conversion-WebUI`

## Step 1 — Ask user

1. **Model A path** (`.pth` trong `assets/weights/` hoặc `logs/*/G_*.pth`)
2. **Model B path** (same)
3. **Blend ratio `alpha`** — float 0.0–1.0
   - `0.0` = 100% B, 0% A
   - `0.5` = 50/50
   - `0.7` = 70% A, 30% B (default — A là dominant voice)
4. **Output name** (no spaces, e.g. `Hybrid_AB`)
5. **Sample rate** — must match cả 2 models (32k/40k/48k)
6. **Has F0?** (`1` for singing-capable models, `0` for speech only)

Nếu user không nhớ models có sẵn:
```powershell
Get-ChildItem assets/weights/ -Filter *.pth | Select-Object Name, Length
```

## Step 2 — Pre-merge validation

**Critical:** merge sẽ fail nếu architectures khác nhau (vd v1 vs v2, 256 vs 768 hidden, khác sample rate).

```python
import torch
ck1 = torch.load(r'MODEL_A_PATH', map_location='cpu', weights_only=False)
ck2 = torch.load(r'MODEL_B_PATH', map_location='cpu', weights_only=False)

# Extract weights
def get_weight(ck):
    if 'model' in ck:
        return {k: v for k, v in ck['model'].items() if 'enc_q' not in k}
    return ck['weight']

w1, w2 = get_weight(ck1), get_weight(ck2)

# Architecture check
keys_match = sorted(w1.keys()) == sorted(w2.keys())
ver1 = ck1.get('version', 'unknown')
ver2 = ck2.get('version', 'unknown')
sr1  = ck1.get('config', [None]*6)[-1] if 'config' in ck1 else 'unknown'
sr2  = ck2.get('config', [None]*6)[-1] if 'config' in ck2 else 'unknown'
print(f"A: version={ver1} sr={sr1} keys={len(w1)}")
print(f"B: version={ver2} sr={sr2} keys={len(w2)}")
print(f"Architectures match: {keys_match}")
```

| Check | Required |
|---|---|
| `keys_match == True` | Layer names identical |
| `version` same | v1↔v1 hoặc v2↔v2 (không cross) |
| `sr` same | 40k↔40k etc. |

**Nếu fail bất kỳ check nào → stop, báo user, không merge.**

## Step 3 — Run merge

Dùng builtin `infer.lib.train.process_ckpt.merge`:
```python
import sys; sys.path.insert(0, '.')
from infer.lib.train.process_ckpt import merge

result = merge(
    path1   = r'MODEL_A_PATH',
    path2   = r'MODEL_B_PATH',
    alpha1  = ALPHA,            # weight of A
    sr      = 'SR',             # '40k' / '48k' / '32k'
    f0      = 1,                # 1 if singing, 0 if speech
    info    = f'Merged from A (alpha={ALPHA}) + B (alpha={1-ALPHA})',
    name    = 'OUTPUT_NAME',
    version = 'v2'              # must match both inputs
)
print(result)  # success message + output path
```

Output sẽ ở `assets/weights/OUTPUT_NAME.pth`.

## Step 4 — Verify merged model

```python
import torch, os
out = f'assets/weights/OUTPUT_NAME.pth'
assert os.path.exists(out), 'merge did not produce output'
size_mb = os.path.getsize(out) / 1024**2
ck = torch.load(out, map_location='cpu', weights_only=False)
print(f"Merged: {out}  ({size_mb:.1f} MB)")
print(f"  keys     : {len(ck['weight'])}")
print(f"  version  : {ck.get('version','?')}")
print(f"  sr       : {ck.get('config',[None]*6)[-1]}")
print(f"  info     : {ck.get('info','?')}")
```

Sanity gate: output size phải ~50–60 MB (extracted small model). Nếu < 30 MB → likely truncated/corrupt.

## Step 5 — Optional: quick inference test

Suggest user test ngay với 1 file ngắn để xem voice có hợp lý không:
```powershell
$INDEX_A = (Get-ChildItem "logs/MODEL_A_NAME/" -Filter "added_*.index" | Select-Object -First 1).FullName
python tools/infer_cli.py `
  --f0up_key 0 `
  --input_path "TEST_AUDIO.wav" `
  --index_path $INDEX_A `
  --opt_path "test_merged.wav" `
  --model_name "OUTPUT_NAME.pth" `
  --index_rate 0.5 `
  --f0method rmvpe
```

**Note về index:** merged model không có index riêng. Dùng tạm index của Model A (hoặc B). Nếu muốn proper index → cần re-extract features từ 1 dataset reference rồi build index mới.

## Step 6 — Final report

```
=== Model Merge Complete ===
Model A      : MODEL_A_NAME      (alpha=X.XX)
Model B      : MODEL_B_NAME      (alpha=Y.YY)
Architecture : v2 / sr=40k / f0=1
Compatible   : YES

Output:
  File : assets/weights/OUTPUT_NAME.pth  (XX.X MB)
  Info : Merged from A (alpha=...) + B (alpha=...)

Limitations:
  - No dedicated FAISS index — reuse A's or B's index
  - Quality empirical: alpha 0.3–0.7 thường ổn; gần 0/1 thì gần model gốc
  - Cross-version (v1+v2) NOT supported
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Fail to merge — architectures not same` | Different version (v1 vs v2) or sr | Re-pick matching models |
| Output < 30 MB | One input was full G/D pair, không phải extracted small | Use `extract_small_model` first via WebUI |
| `UnpicklingError` | PyTorch 2.6 + old fairseq | Already loads with `weights_only=False` — should not occur, but if it does, see CLAUDE.md known bugs |
| Hybrid voice sounds robotic | alpha too extreme | Try alpha closer to 0.5 |
