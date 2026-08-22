"""Generate notebooks/ml/01_train_oaizib_colab.ipynb.

Written as a generator rather than hand-authored JSON so the notebook stays
diffable and cannot drift into invalid nbformat.

The notebook is deliberately **self-contained**: it does not clone the repo. The
repo's only committed state is hours old, and a notebook that depends on an
uncommitted working tree is a notebook that fails on someone else's machine. It
therefore reimplements the data bridge and training loop, but writes its
checkpoint in the *exact* format ``kneeai.ml.infer`` expects, so the artefact
drops straight into the platform with no conversion step.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks/ml/01_train_oaizib_colab.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(
    """
# OAIZIB-CM knee segmentation — Colab training

Trains the bone + cartilage segmentation model for the Knee Intelligence Platform.

**Runtime → Change runtime type → T4 GPU** before running anything.

### What this produces
`checkpoint.pt` in the exact format `kneeai.ml.infer` loads, plus `metrics.json`
with per-structure Dice, IoU, HD95 and ASSD. Download both at the end and drop
them into `artifacts/models/seg_oaizib/`.

### Why it is not being run on the laptop
The dev machine is an 8 GB M1. An SSL-pretrained SwinUNETR at 96³ does not fit in
unified memory, and the disk has little room spare. Colab has both the VRAM and
the disk.

### Scope
**150 training cases**, agreed as sufficient. Only those 150 images are
downloaded — range-fetched from the archive at ~3.7 GB rather than 9.3 GB. Model
architecture, resolution and augmentation are unchanged from the full-scale
recipe; only the number of cases is smaller.

### The problem this run is fixing
A previous 30-patient run reached femur 0.913 and tibia 0.890 but only
**0.455 / 0.423** on the tibial cartilages. Those two structures are 0.1% of all
voxels. Three changes target that: class-balanced crop sampling, cross-entropy
weights measured from the data, and far more training data.

### Honesty constraints baked in
- The validation split is **patient-level** and asserted to share zero patients
  with training.
- Upstream's `imagesTr`/`imagesTs` boundary is respected; the held-out material
  is never used for model selection.
- Every OAIZIB-CM knee is a **right** knee, so sagittal mirroring is mandatory or
  the model never sees a left knee in any form.
- OAIZIB-CM has **no meniscus and no patella labels**. This model cannot produce
  them and the metrics will not pretend otherwise.
- Licence is **CC-BY-NC-4.0**: non-commercial, and citing Yao et al. (CartiMorph,
  *Medical Image Analysis* 91:103035, 2024) is compulsory.
"""
)

# ---------------------------------------------------------------------------
md("## 1. Configuration\n\nPick a profile. Read the trade-off before choosing.")

code(
    '''
# ============================================================================
# PROFILE: "standard" | "fallback"
# ----------------------------------------------------------------------------
#   standard  ~2 h   150 cases, SSL-pretrained SwinUNETR, 0.7 mm, 96^3 patches.
#                    150 was agreed as sufficient, so the images are RANGE-FETCHED
#                    out of imagesTr.zip: ~3.7 GB instead of the full 9.3 GB.
#                    Nothing about model quality is reduced - only the number of
#                    cases changed.
#
#   fallback  ~40 m  103 cases from the smaller test archive, 3D UNet, 1.0 mm.
#                    Use only if range requests fail or the session is unstable.
#                    1.0 mm is what caused the original cartilage failure, so this
#                    is a safety net and not a substitute.
# ============================================================================
PROFILE = "standard"

CONFIG = {
    "standard": dict(
        archive="train", limit_cases=150, range_fetch=True,
        arch="swin_unetr", spacing=0.7, patch=96, batch=2, samples=2,
        epochs=60, lr=2e-4, val_every=4, feature_size=48, pretrained=True,
        # 150 cases x 2 crops / batch 2 = 150 steps for a full pass, roughly
        # 90 s/epoch on a T4. Capping iterations per epoch decouples "how long is
        # an epoch" from "how much data exists" - what nnU-Net does, for the same
        # reason. Every case is still reached: the sampler draws randomly across
        # all 150 each epoch, so over 60 epochs each is seen many times.
        steps_per_epoch=150,
    ),
    "fallback": dict(
        archive="test", limit_cases=103, range_fetch=False,
        arch="unet", spacing=1.0, patch=64, batch=2, samples=2,
        epochs=60, lr=1e-3, val_every=5,
        features=(16, 32, 64, 128, 256), pretrained=False,
        steps_per_epoch=None,
    ),
}[PROFILE]

VAL_FRACTION = 0.20
SEED = 42

# Persist to Drive so a dropped session does not lose the run.
USE_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/kneeai_training"

print(f"profile: {PROFILE}")
for k, v in CONFIG.items():
    print(f"  {k:14} {v}")

# Rough wall-clock estimate so a run that cannot finish is obvious now rather
# than in four hours. Per-step cost is measured for real after epoch 1.
est_per_step = 0.60 if CONFIG["arch"] == "swin_unetr" else 0.22
est_steps = CONFIG.get("steps_per_epoch") or (
    CONFIG["limit_cases"] * CONFIG["samples"] // CONFIG["batch"]
)
est_hours = CONFIG["epochs"] * est_steps * est_per_step / 3600
print("")
print(f"rough estimate: {est_steps} steps/epoch, ~{est_hours:.1f} h training")
print("(plus download and caching; the real figure is printed after epoch 1)")
if est_hours > 6:
    print("WARNING: longer than a free Colab session usually survives.")
'''
)

# ---------------------------------------------------------------------------
md("## 2. Environment\n\nRecorded into the model card later, so capture it verbatim.")

code(
    '''
import subprocess, sys, json, os, platform

print("=" * 68)
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("nvidia-smi unavailable:", e)
    print("\\n*** NO GPU. Runtime -> Change runtime type -> T4 GPU, then rerun. ***")

!pip -q install "monai==1.5.*" einops nibabel huggingface_hub remotezip psutil openpyxl 2>&1 | tail -2

import torch, monai
ENV = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "monai": monai.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    if torch.cuda.is_available() else None,
}
print(json.dumps(ENV, indent=2))
assert torch.cuda.is_available(), "GPU required"
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 3. Download OAIZIB-CM

Ungated on HuggingFace — no token, no DUA. Pinned to a revision so the run is
reproducible.

**Only the cases we train on are downloaded.** A zip's central directory sits at
the end of the file, so with HTTP range requests the member list can be read and
individual entries pulled without fetching the archive. For 150 cases that is
~3.7 GB rather than the full 9.3 GB.

Labels come down whole — `labelsTr.zip` is 55 MB and not worth ranging.

The 150 are chosen by a seeded shuffle over the sorted member list, so the subset
is reproducible and is not just the numerically-lowest case ids.
"""
)

code(
    '''
import random
import re
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "YongchengYAO/OAIZIB-CM"
REVISION = "d2af7abff1e9bb85212bdd70e11d60bbb21939fa"   # pinned
ROOT = Path("/content/data/oaizib"); ROOT.mkdir(parents=True, exist_ok=True)

# 454 was flagged corrupt upstream on 14-Feb-2025 and replaced the next day.
# Excluded by default; the file may well be fine now.
EXCLUDE_CASES = {454}

if CONFIG["archive"] == "train":
    IMAGES_ZIP, LABELS_ZIP, IMG_DIR, LBL_DIR = "imagesTr.zip", "labelsTr.zip", "imagesTr", "labelsTr"
else:
    IMAGES_ZIP, LABELS_ZIP, IMG_DIR, LBL_DIR = "imagesTs.zip", "labelsTs.zip", "imagesTs", "labelsTs"

t0 = time.time()

# --- labels + info: small, fetch whole -------------------------------------
for name in (LABELS_ZIP, "info.zip"):
    print(f"downloading {name} ...", flush=True)
    path = hf_hub_download(REPO_ID, name, repo_type="dataset",
                           revision=REVISION, local_dir="/content/dl")
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
    print(f"  extracted  ({time.time()-t0:.0f}s)")

# --- images ----------------------------------------------------------------
IMAGES_URL = (f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
              f"{REVISION}/{IMAGES_ZIP}")

def case_number(name):
    m = re.search(r"(\\d+)", Path(name).name)
    return int(m.group(1)) if m else None

fetched = 0
if CONFIG["range_fetch"]:
    from remotezip import RemoteZip

    print(f"\\nrange-fetching {CONFIG['limit_cases']} cases from {IMAGES_ZIP}")
    print("(reading the central directory, not the archive)")
    (ROOT / IMG_DIR).mkdir(parents=True, exist_ok=True)

    with RemoteZip(IMAGES_URL) as rz:
        members = [n for n in rz.namelist() if n.endswith(".nii.gz")]
        members = [n for n in members if case_number(n) not in EXCLUDE_CASES]
        members.sort()
        # Seeded shuffle: reproducible, and not just the lowest case ids.
        random.Random(SEED).shuffle(members)
        chosen = members[: CONFIG["limit_cases"]]
        print(f"archive holds {len(members)} usable images; taking {len(chosen)}")

        for i, member in enumerate(chosen, 1):
            target = ROOT / IMG_DIR / Path(member).name
            if target.exists():
                fetched += 1
                continue
            with rz.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            fetched += 1
            if i % 10 == 0 or i == len(chosen):
                mb = sum(f.stat().st_size for f in (ROOT / IMG_DIR).glob("*.nii.gz")) / 1e6
                print(f"  {i}/{len(chosen)}  {mb:,.0f} MB  ({time.time()-t0:.0f}s)",
                      flush=True)
else:
    print(f"\\ndownloading {IMAGES_ZIP} whole ...", flush=True)
    path = hf_hub_download(REPO_ID, IMAGES_ZIP, repo_type="dataset",
                           revision=REVISION, local_dir="/content/dl")
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
    fetched = len(list((ROOT / IMG_DIR).glob("*.nii.gz")))

n_img = len(list((ROOT / IMG_DIR).glob("*.nii.gz")))
n_lbl = len(list((ROOT / LBL_DIR).glob("*.nii.gz")))
size_gb = sum(f.stat().st_size for f in ROOT.rglob("*.nii.gz")) / 1e9

print(f"\\nimages on disk : {n_img}")
print(f"labels on disk : {n_lbl}")
print(f"total size     : {size_gb:.2f} GB")
print(f"elapsed        : {time.time()-t0:.0f}s")
assert n_img > 0, "no images downloaded"
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 4. Manifest and patient-level split

Two things that decide whether the reported Dice means anything:

**Patient-level grouping.** OAI is longitudinal. Splitting by *case* puts the same
knee on both sides of the split and inflates the score with nothing in the log to
show it. Asserted here, not assumed.

**KL stratification.** Cartilage is exactly what degrades with Kellgren-Lawrence
grade. At this sample size an unstratified draw can put most of the severe-OA
knees on one side, and the validation score would then reflect the draw as much
as the model.
"""
)

code(
    '''
import re, random, collections
import pandas as pd

LABEL_MAP = {
    "background": 0, "femur": 1, "femoral_cartilage": 2, "tibia": 3,
    "medial_tibial_cartilage": 4, "lateral_tibial_cartilage": 5,
}
STRUCTURES = [k for k, v in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]) if v != 0]
N_CLASSES = len(LABEL_MAP)

# --- subject metadata (patient id, KL grade, demographics) ---
info_dir = next((d for d in ROOT.rglob("*") if d.is_dir() and
                 any(f.name.startswith("subInfo") for f in d.iterdir())), ROOT)
sub_files = sorted(info_dir.glob("subInfo*"))
print("subject info files:", [f.name for f in sub_files])

meta = {}
for f in sub_files:
    df = pd.read_excel(f) if f.suffix in (".xlsx", ".xls") else pd.read_csv(f)
    cols = {c.lower().strip(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    c_case = col("cmtid", "id", "caseid", "case")
    c_pat  = col("subjectid", "patientid", "subject", "patient")
    c_kl   = col("klgrade", "kl", "grade")
    for _, r in df.iterrows():
        try: cid = int(r[c_case])
        except Exception: continue
        meta[cid] = {
            "patient_id": str(r[c_pat]) if c_pat else f"unknown-{cid}",
            "kl_grade": (int(r[c_kl]) if c_kl and pd.notna(r[c_kl]) else None),
        }
print(f"subjects with metadata: {len(meta)}")

# --- scan cases ---
# Index the labels by case NUMBER rather than by constructed filename. The train
# archive zero-pads (oaizib_001.nii.gz) while the test archive does not
# (oaizib_405.nii.gz), so building the name from an int silently pairs nothing on
# one of the two.
labels_by_case = {}
for lbl in (ROOT / LBL_DIR).glob("*.nii.gz"):
    cid = case_number(lbl.name)
    if cid is not None:
        labels_by_case[cid] = lbl
print(f"labels indexed: {len(labels_by_case)}")

records = []
missing_label = 0
for img in sorted((ROOT / IMG_DIR).glob("*.nii.gz")):
    cid = case_number(img.name)
    if cid is None or cid in EXCLUDE_CASES: continue
    lbl = labels_by_case.get(cid)
    if lbl is None:
        missing_label += 1
        continue
    m = meta.get(cid, {"patient_id": f"unknown-{cid}", "kl_grade": None})
    records.append({
        "case_id": f"oaizib-{cid}", "cmt_id": cid,
        "patient_id": m["patient_id"], "kl_grade": m["kl_grade"],
        "image": str(img), "label": str(lbl),
        "side": "right",     # OAIZIB-CM is 507 right knees, zero left
        "modality": "MR",
    })

random.Random(SEED).shuffle(records)
records = records[: CONFIG["limit_cases"]]
print(f"cases usable: {len(records)}")
if missing_label:
    print(f"  {missing_label} image(s) had no matching label and were skipped")
assert records, (
    "no image/label pairs found. Check that both archives extracted and that "
    "case numbers line up between them."
)

# --- patient-level, KL-stratified split ---
by_patient = collections.defaultdict(list)
for r in records:
    by_patient[r["patient_id"]].append(r)

buckets = collections.defaultdict(list)
for pid, rs in by_patient.items():
    buckets[rs[0]["kl_grade"]].append(pid)

rng = random.Random(SEED)
val_patients = set()
for grade, pids in buckets.items():
    pids = sorted(pids); rng.shuffle(pids)
    k = max(1, round(len(pids) * VAL_FRACTION))
    val_patients.update(pids[:k])

train = [r for r in records if r["patient_id"] not in val_patients]
val   = [r for r in records if r["patient_id"] in val_patients]

tp = {r["patient_id"] for r in train}; vp = {r["patient_id"] for r in val}
assert not (tp & vp), f"PATIENT LEAKAGE: {tp & vp}"
print(f"train {len(train)} cases / {len(tp)} patients")
print(f"val   {len(val)} cases / {len(vp)} patients")
print("LEAKAGE CHECK: PASS (0 shared patients)")

def kl_hist(rs):
    c = collections.Counter(str(r["kl_grade"]) for r in rs)
    return dict(sorted(c.items()))
print("KL train:", kl_hist(train))
print("KL val  :", kl_hist(val))
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 5. Preprocessing and augmentation

**Class-balanced cropping is the key change.** The previous run used
`RandCropByPosNegLabeld`, which treats every non-background voxel as equally
worth sampling. Femur is 1.86 M voxels; medial tibial cartilage is 19.5 k. So
roughly one crop in a hundred was centred on the structure that was failing.
`RandCropByLabelClassesd` with explicit ratios forces the sampler onto cartilage.

**Sagittal mirroring is not optional.** Every training knee is a right knee.
Medial and lateral are *anatomical* rather than spatial designations, so mirroring
across the sagittal plane produces a correctly labelled left knee. Without it the
model has never seen half of all patients.
"""
)

code(
    '''
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd,
    RandCropByLabelClassesd, RandFlipd, RandGaussianNoised,
    RandShiftIntensityd, RandAdjustContrastd, RandRotate90d,
)
from monai.utils import set_determinism
set_determinism(seed=SEED)

SPACING = CONFIG["spacing"]; PATCH = CONFIG["patch"]

# background, femur, fem cart, tibia, med tib cart, lat tib cart
CROP_RATIOS = [1, 2, 4, 2, 5, 5]

base = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(SPACING,)*3, mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5,
                                   b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys="image", nonzero=True),
]

train_tf = Compose(base + [
    RandCropByLabelClassesd(
        keys=["image", "label"], label_key="label",
        spatial_size=(PATCH,)*3, ratios=CROP_RATIOS,
        num_classes=N_CLASSES, num_samples=CONFIG["samples"], allow_smaller=True,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # right -> left knee
    RandGaussianNoised(keys="image", prob=0.15, std=0.03),
    RandShiftIntensityd(keys="image", prob=0.20, offsets=0.10),
    RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.8, 1.3)),
])
val_tf = Compose(base)
print(f"resample {SPACING} mm iso  patch {PATCH}^3  crop ratios {CROP_RATIOS}")
'''
)

code(
    '''
# --- cross-entropy class weights, measured from the training labels ---
import numpy as np, nibabel as nib

counts = np.zeros(N_CLASSES)
for r in train[: min(20, len(train))]:
    a = np.asarray(nib.load(r["label"]).dataobj)
    for c in range(N_CLASSES):
        counts[c] += float((a == c).sum())

freq = counts / counts.sum()
w = 1.0 / np.maximum(freq, 1e-8)
w = w / w[1]                 # femur == 1.0
w = np.clip(w, 0.2, 25.0)    # uncapped inverse frequency destabilises early training
w[0] = 0.2                   # Dice excludes background anyway

print("voxel fraction  :", dict(zip(["bg"] + STRUCTURES, np.round(freq, 5))))
print("CE class weights:", dict(zip(["bg"] + STRUCTURES, np.round(w, 2))))
CLASS_WEIGHTS = tuple(float(x) for x in w)
'''
)

md(
    """
### Caching to disk, not RAM

A 0.7 mm volume is 160 x 200 x 200 = 6.4 M voxels, so image plus label is roughly
50 MB per case once transformed. Holding 150 of those in RAM is ~7.6 GB, and Colab
gives about 12.7 GB — which looks survivable until DataLoader workers are added.

Forked workers do not share that memory cheaply. MONAI `MetaTensor` carries Python
objects, and refcount updates touch the pages, so copy-on-write collapses and each
worker duplicates much of the cache. The result is
`DataLoader worker is killed by signal: Killed` — SIGKILL from the host OOM killer,
not a GPU error.

So this uses **`PersistentDataset`**: the deterministic transforms run once and the
result is written to local disk, which Colab has ~100 GB of. RAM then holds only
paths and hashes, forking workers is cheap again, and epoch two onward reads
pre-transformed tensors straight off disk.

The cache goes to `/content`, deliberately not Drive. Drive is network-backed and
would make every batch a round trip.
"""
)

code(
    '''
import shutil
import time
from pathlib import Path

import psutil
from monai.data import DataLoader, PersistentDataset

# Fail with a useful message if an earlier cell did not finish, rather than a
# bare NameError forty lines down.
for required in ("train", "val", "train_tf", "val_tf", "CONFIG"):
    if required not in globals():
        raise RuntimeError(f"{required!r} is undefined - re-run the earlier cells in order.")

def files(rs):
    return [{"image": r["image"], "label": r["label"],
             "case_id": r["case_id"], "patient_id": r["patient_id"]} for r in rs]

# Local disk, NOT Drive: Drive is network-backed and would make every batch a
# round trip.
CACHE_DIR = Path("/content/cache")
free_gb = shutil.disk_usage("/content").free / 1e9
print(f"RAM total  : {psutil.virtual_memory().total/1e9:.1f} GB")
print(f"disk free  : {free_gb:.1f} GB at /content")
print(f"cache dir  : {CACHE_DIR}")

# ~50 MB per transformed case at 0.7 mm; the check is deliberately generous.
need_gb = len(train + val) * 0.06
if free_gb < need_gb + 5:
    raise RuntimeError(
        f"need roughly {need_gb:.1f} GB of cache space but only {free_gb:.1f} GB free"
    )

t0 = time.time()
train_ds = PersistentDataset(files(train), train_tf, cache_dir=CACHE_DIR / "train")
val_ds = PersistentDataset(files(val), val_tf, cache_dir=CACHE_DIR / "val")
print(f"datasets ready in {time.time()-t0:.1f}s")
print("(transforms run and are written to disk on first access, so epoch 1 is")
print(" slower than the rest; epoch 2 onward reads pre-transformed tensors)")

# Workers are safe again: the parent holds only paths, so forking is cheap and
# each worker reads one item at a time from the disk cache.
train_dl = DataLoader(train_ds, batch_size=CONFIG["batch"], shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True,
                      persistent_workers=True, prefetch_factor=2)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

FULL_PASS = len(train_dl)
STEPS_PER_EPOCH = min(CONFIG.get("steps_per_epoch") or FULL_PASS, FULL_PASS)
print(f"\\nfull pass   : {FULL_PASS} steps")
print(f"steps/epoch : {STEPS_PER_EPOCH}")
if STEPS_PER_EPOCH < FULL_PASS:
    print(f"  capped at {STEPS_PER_EPOCH * CONFIG['batch']} crops per epoch, drawn at "
          f"random across all {len(train)} cases. Over {CONFIG['epochs']} epochs "
          f"every case is sampled many times over.")

print(f"RAM used now: {psutil.virtual_memory().percent:.0f}%  "
      f"(should stay flat during training)")
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 6. Model

For the `full` profile this loads MONAI's **self-supervised pretrained SwinUNETR
encoder** (trained on roughly 50k 3D CT and MR volumes) and fine-tunes it. That is
the pretrain → fine-tune pipeline, finally on hardware that can hold it.

The cell **prints how many tensors actually matched**. A silent partial load is the
most common way to end up with a "pretrained" model that is in fact random, and it
never announces itself — the loss curve just looks slightly worse and nobody knows
why.
"""
)

code(
    '''
import torch, torch.nn as nn
from monai.networks.nets import UNet, SwinUNETR

device = torch.device("cuda")

if CONFIG["arch"] == "unet":
    model = UNet(spatial_dims=3, in_channels=1, out_channels=N_CLASSES,
                 channels=CONFIG["features"], strides=(2,)*(len(CONFIG["features"])-1),
                 num_res_units=2, norm="instance").to(device)
    MODEL_CONFIG_EXTRA = {"features": list(CONFIG["features"])}
    ENCODER_REPORT = {"pretrained": False, "reason": "UNet has no published SSL checkpoint"}
else:
    try:
        model = SwinUNETR(img_size=(PATCH,)*3, in_channels=1, out_channels=N_CLASSES,
                          feature_size=CONFIG["feature_size"], use_checkpoint=True).to(device)
    except TypeError:
        # img_size was deprecated in newer MONAI
        model = SwinUNETR(in_channels=1, out_channels=N_CLASSES,
                          feature_size=CONFIG["feature_size"], use_checkpoint=True).to(device)
    MODEL_CONFIG_EXTRA = {"feature_size": CONFIG["feature_size"]}

    ENCODER_REPORT = {"pretrained": False}
    if CONFIG["pretrained"]:
        URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/"
               "releases/download/0.8.1/model_swinvit.pt")
        try:
            # urlretrieve rather than !wget: a shell escape returns a non-zero exit
            # code instead of raising, so a failed download would slip past the
            # except block and be reported as a successful load of nothing.
            import urllib.request
            urllib.request.urlretrieve(URL, "/content/model_swinvit.pt")
            sd = torch.load("/content/model_swinvit.pt", map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", "").replace("swinViT.", "").replace("swin_vit.", ""): v
                  for k, v in sd.items()}
            target = model.swinViT.state_dict()
            usable = {k: v for k, v in sd.items()
                      if k in target and target[k].shape == v.shape}
            missing, unexpected = model.swinViT.load_state_dict(usable, strict=False)
            ENCODER_REPORT = {
                "pretrained": True,
                "source": "MONAI SSL SwinViT (~50k 3D CT+MR volumes)",
                "url": URL,
                "tensors_in_checkpoint": len(sd),
                "tensors_matched": len(usable),
                "tensors_in_encoder": len(target),
                "missing": len(missing),
                "unexpected": len(unexpected),
            }
        except Exception as e:
            ENCODER_REPORT = {"pretrained": False, "reason": f"download/load failed: {e}"}

n_params = sum(p.numel() for p in model.parameters())
print(f"{CONFIG['arch']}  {n_params/1e6:.2f} M params  out_channels={N_CLASSES}")
print("encoder:", json.dumps(ENCODER_REPORT, indent=2))
if ENCODER_REPORT.get("pretrained") and ENCODER_REPORT["tensors_matched"] == 0:
    print("\\n*** WARNING: 0 tensors matched. This is NOT pretrained. ***")
'''
)

code(
    '''
from monai.losses import DiceCELoss
import inspect

# MONAI renamed ce_weight -> weight. Pick whatever this version accepts rather
# than letting the weights be silently dropped.
wkw = "weight" if "weight" in inspect.signature(DiceCELoss.__init__).parameters else "ce_weight"
loss_fn = DiceCELoss(
    to_onehot_y=True, softmax=True, include_background=False,
    **{wkw: torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)},
    lambda_dice=1.0, lambda_ce=1.0,
)
opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
scaler = torch.amp.GradScaler("cuda")
print(f"loss DiceCE (weights via '{wkw}')   opt AdamW lr={CONFIG['lr']}   AMP on")
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 7. Train

Prints train loss every epoch and validation Dice every `val_every`. Checkpoints
the best model to Drive, so a dropped session resumes rather than restarts.
"""
)

code(
    '''
import psutil
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path(DRIVE_DIR); OUT.mkdir(parents=True, exist_ok=True)
else:
    OUT = Path("/content/out"); OUT.mkdir(parents=True, exist_ok=True)
print("output dir:", OUT)

post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=N_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=N_CLASSES)])
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

RESUME = OUT / "last.pt"
start_epoch, best, history = 1, -1.0, []
if RESUME.exists():
    ck = torch.load(RESUME, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
    start_epoch = ck["epoch"] + 1; best = ck["best"]; history = ck["history"]
    print(f"resumed from epoch {ck['epoch']} (best {best:.4f})")

for epoch in range(start_epoch, CONFIG["epochs"] + 1):
    model.train(); t0 = time.time(); run = n = 0
    for batch in train_dl:
        if n >= STEPS_PER_EPOCH:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run += float(loss.item()); n += 1
        print(f"  ep{epoch:03d} step {n:03d}/{STEPS_PER_EPOCH} loss {loss.item():.4f}",
              end="\\r", flush=True)
    sched.step()

    # ETA is measured on the SECOND epoch. The first one also writes the disk
    # cache, so timing it would overstate the remaining run by a wide margin.
    if epoch == start_epoch + 1:
        secs = time.time() - t0
        remaining = CONFIG["epochs"] - epoch
        n_val = max(0, remaining // CONFIG["val_every"])
        total_h = (secs * remaining + n_val * secs * 0.4) / 3600
        print(" " * 118, end="\\r")
        print(f"MEASURED: {secs:.0f}s/epoch at {secs/max(n,1):.2f}s/step "
              f"-> ~{total_h:.1f} h for the remaining {remaining} epochs")
        print(f"          RAM {psutil.virtual_memory().percent:.0f}%  "
              f"VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak")
        if total_h > 5:
            print("  Longer than a free Colab session usually survives. Interrupt and")
            print("  lower CONFIG['epochs'] or steps_per_epoch; last.pt lets you resume.")
        print()
    row = {"epoch": epoch, "train_loss": run / max(n, 1), "seconds": time.time() - t0}
    line = (f"epoch {epoch:03d}/{CONFIG['epochs']}  train_loss {row['train_loss']:.4f}  "
            f"lr {opt.param_groups[0]['lr']:.2e}  {row['seconds']:.0f}s")

    if epoch % CONFIG["val_every"] == 0 or epoch == CONFIG["epochs"]:
        model.eval(); dice_metric.reset()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in val_dl:
                x = batch["image"].to(device); y = batch["label"].to(device)
                logits = sliding_window_inference(x, (PATCH,)*3, 2, model, overlap=0.25)
                dice_metric(y_pred=[post_pred(p) for p in decollate_batch(logits.float())],
                            y=[post_label(l) for l in decollate_batch(y)])
        per = dice_metric.aggregate().cpu().numpy()
        mean = float(np.nanmean(per))
        row["val_dice_mean"] = mean
        row["val_dice_per_structure"] = {s: float(d) for s, d in zip(STRUCTURES, per)}
        line += f"  |  dice {mean:.4f}  [" + " ".join(
            f"{s.split('_')[0][:4]}={d:.3f}" for s, d in zip(STRUCTURES, per)) + "]"
        if mean > best:
            best = mean
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_dice_mean": mean,
                        "val_dice_per_structure": row["val_dice_per_structure"]},
                       OUT / "best.pt")
            line += "  <- saved"

    print(" " * 118, end="\\r"); print(line, flush=True)
    history.append(row)
    torch.save({"model_state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "history": history}, OUT / "last.pt")
    (OUT / "history.json").write_text(json.dumps(history, indent=2))

print("-" * 68)
print(f"best val mean Dice: {best:.4f}")
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 8. Full evaluation

Dice alone is not enough (PRD §114). Hausdorff-95 and average symmetric surface
distance are what actually predict whether a mesh is usable for measurement: a
mask can score a respectable Dice while carrying a spur that moves a landmark
several millimetres.
"""
)

code(
    '''
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric

ck = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model_state_dict"]); model.eval()
print(f"evaluating best checkpoint (epoch {ck['epoch']}, dice {ck['val_dice_mean']:.4f})")

metrics = {
    "dice": DiceMetric(include_background=False, reduction="mean_batch"),
    "iou":  MeanIoU(include_background=False, reduction="mean_batch"),
    "hd95": HausdorffDistanceMetric(include_background=False, percentile=95,
                                    reduction="mean_batch"),
    "assd": SurfaceDistanceMetric(include_background=False, symmetric=True,
                                   reduction="mean_batch"),
}
for m in metrics.values(): m.reset()

with torch.no_grad(), torch.amp.autocast("cuda"):
    for i, batch in enumerate(val_dl, 1):
        x = batch["image"].to(device); y = batch["label"].to(device)
        logits = sliding_window_inference(x, (PATCH,)*3, 2, model, overlap=0.5)
        p = [post_pred(q) for q in decollate_batch(logits.float())]
        g = [post_label(q) for q in decollate_batch(y)]
        for m in metrics.values(): m(y_pred=p, y=g)
        print(f"  {i}/{len(val_dl)}", end="\\r")

RESULTS = {}
for name, m in metrics.items():
    vals = m.aggregate().cpu().numpy()
    RESULTS[name] = {s: (None if not np.isfinite(v) else round(float(v), 4))
                     for s, v in zip(STRUCTURES, vals)}

print("\\n" + "=" * 78)
print(f"{'structure':28} {'Dice':>8} {'IoU':>8} {'HD95(mm)':>10} {'ASSD(mm)':>10}")
print("-" * 78)
for s in STRUCTURES:
    def fmt(k):
        v = RESULTS[k][s]
        return "  n/a" if v is None else f"{v:8.4f}"
    print(f"{s:28} {fmt('dice')} {fmt('iou')} {fmt('hd95')} {fmt('assd')}")
print("-" * 78)
mean_dice = float(np.nanmean([v for v in RESULTS['dice'].values() if v is not None]))
print(f"{'mean (excl. background)':28} {mean_dice:8.4f}")
print("=" * 78)
print("\\nNOTE: HD95 and ASSD are in millimetres at the inference spacing")
print(f"      ({SPACING} mm isotropic), not in voxels.")
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 9. Save in the platform's checkpoint format

`kneeai.ml.infer` is strict about three keys, and each strictness exists for a
reason:

- `model_state_dict` — a nested `state_dict` gets read as though the metadata
  entries were parameter tensors.
- `label_map` — **must include background**, because `out_channels` is derived
  from `len(label_map)`.
- `model_config` — without it the predictor rebuilds with default
  `features=(32, 64, 128, 256, 320)` and every tensor shape mismatches.

Getting these right here means the download drops straight in with no conversion.
"""
)

code(
    '''
MODEL_CONFIG = {
    "architecture": CONFIG["arch"],
    "in_channels": 1,
    "out_channels": N_CLASSES,
    "patch_size": [PATCH, PATCH, PATCH],
    "spatial_dims": 3,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "loss": "dice_ce",
    "include_background_in_loss": False,
    "class_weights": list(CLASS_WEIGHTS),
    "optimizer": "adamw",
    "learning_rate": CONFIG["lr"],
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "max_epochs": CONFIG["epochs"],
    **MODEL_CONFIG_EXTRA,
}

payload = {
    "model_state_dict": ck["model_state_dict"],
    "label_map": LABEL_MAP,             # includes background, on purpose
    "model_config": MODEL_CONFIG,
    "epoch": ck["epoch"],
    "val_dice_mean": mean_dice,
    "val_dice_per_structure": RESULTS["dice"],
    "structures": STRUCTURES,
    "n_classes": N_CLASSES,
    "spacing_mm": [SPACING] * 3,
    "patch": PATCH,
    "run": f"colab-{PROFILE}",
}
torch.save(payload, OUT / "checkpoint.pt")

REPORT = {
    "profile": PROFILE,
    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()},
    "environment": ENV,
    "encoder": ENCODER_REPORT,
    "dataset": {
        "dataset_id": "oaizib_cm",
        "repo": REPO_ID, "revision": REVISION,
        "archive": CONFIG["archive"],
        "n_cases": len(records),
        "n_patients": len(by_patient),
        "split_patients": {"train": len(tp), "val": len(vp)},
        "split_strategy": "patient_level_stratified_by_kl_grade",
        "patient_leakage": "none (asserted)",
        "kl_histogram": {"train": kl_hist(train), "val": kl_hist(val)},
        "excluded_cases": sorted(EXCLUDE_CASES),
        "sides": ["right"],
        "licence": "CC-BY-NC-4.0",
        "commercial_use_permitted": False,
        "citation_required": ("Yao Y, Zhong J, Zhang L, Khan S, Chen W. CartiMorph. "
                              "Medical Image Analysis 91:103035, 2024. "
                              "doi:10.1016/j.media.2023.103035"),
        "structures_unavailable": ["medial_meniscus", "lateral_meniscus",
                                   "patella", "patellar_cartilage"],
    },
    "preprocessing": {
        "orientation": "RAS",
        "target_spacing_mm": [SPACING] * 3,
        "intensity": "percentile clip 0.5-99.5 then z-score on non-zero voxels",
        "patch_size": [PATCH] * 3,
        "crop_ratios": CROP_RATIOS,
        "class_weights": list(CLASS_WEIGHTS),
        "mirror_augmentation": "sagittal RandFlip p=0.5 (all training knees are right)",
    },
    "metrics": {
        "best_epoch": ck["epoch"],
        "epochs_trained": max(r["epoch"] for r in history),
        "mean_dice_excl_background": mean_dice,
        "per_structure": RESULTS,
        "metric_definition": ("MONAI DiceMetric / MeanIoU / HausdorffDistanceMetric"
                              "(percentile=95) / SurfaceDistanceMetric(symmetric), "
                              "include_background=False, sliding window "
                              f"{PATCH}^3 overlap 0.5, distances in mm"),
    },
    "history": history,
    "limitations": [
        "Research model. Not clinically validated and not for clinical use.",
        "No meniscus or patella output: OAIZIB-CM has no such labels.",
        "All training knees are right knees; left-knee performance is unmeasured "
        "and depends entirely on mirror augmentation.",
        "Single train/val split, no cross-validation, so the reported Dice has a "
        "wide and unquantified confidence interval.",
        "No external validation on any second dataset.",
        "OAI-derived US cohort; not representative of an Indian surgical population.",
        "CC-BY-NC-4.0 training data: this model may not be used commercially.",
    ],
}
(OUT / "metrics.json").write_text(json.dumps(REPORT, indent=2, default=str))

print("wrote:")
for f in ("checkpoint.pt", "metrics.json", "history.json"):
    p = OUT / f
    print(f"  {f:16} {p.stat().st_size/1e6:8.2f} MB")
'''
)

code(
    '''
# Sanity check: does it load the way the platform will load it?
probe = torch.load(OUT / "checkpoint.pt", map_location="cpu", weights_only=False)
assert "model_state_dict" in probe, "missing model_state_dict"
assert probe["label_map"]["background"] == 0, "label_map must include background=0"
assert len(probe["label_map"]) == probe["model_config"]["out_channels"], \\
    "len(label_map) must equal out_channels"
print("checkpoint format: OK")
print("  tensors     :", len(probe["model_state_dict"]))
print("  label_map   :", probe["label_map"])
print("  out_channels:", probe["model_config"]["out_channels"])
print("  architecture:", probe["model_config"]["architecture"])
print("  mean dice   :", round(probe["val_dice_mean"], 4))
'''
)

code(
    '''
from google.colab import files
import shutil
shutil.make_archive("/content/seg_oaizib", "zip", OUT)
print("downloading seg_oaizib.zip ...")
files.download("/content/seg_oaizib.zip")
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 10. Back on the laptop

```bash
unzip seg_oaizib.zip -d artifacts/models/seg_oaizib/

# regenerate the model card from the real metrics
.venv/bin/python scripts/write_model_card.py

# run the full chain on a real patient
.venv/bin/python -m kneeai.cli infer \\
    --case-id oaizib-417 \\
    --source ai_prediction \\
    --image data/raw/oaizib/imagesTs/oaizib_417_0000.nii.gz \\
    --side right --strict
```

That produces the complete case payload — segmentation, meshes, landmarks,
measurements, implant candidates, surgical plan, simulation — and the viewer
badge flips from `PHANTOM` to `AI PREDICTION` with no code change.

### If the session drops
Re-run every cell. The training cell resumes from `last.pt` on Drive.

### `DataLoader worker is killed by signal: Killed`

That is SIGKILL from the host OOM killer — system RAM, not GPU. It should not
happen now that transforms are cached to disk rather than held in memory, but if it
does, in this order:

1. Set `num_workers=0` in **both** DataLoaders. With a disk cache the workers buy
   little, since the expensive transforms already ran.
2. Drop `batch` from 2 to 1.
3. Drop `patch` from 96 to 80.

Do **not** raise `spacing` back to 1.0 mm. That is precisely what caused the
cartilage failure this run exists to fix.

### CUDA out of memory
Different error, different fix: that one is GPU. Lower `patch` first (96 → 80 → 64),
then `batch`. `use_checkpoint=True` is already on, which trades compute for VRAM.

### If range requests fail
Some networks or proxies do not honour HTTP `Range`. Set `PROFILE = "fallback"`,
which downloads a whole smaller archive instead. Note it also drops to 1.0 mm and
a plain UNet, so it is a safety net rather than an equivalent run.

### If a cell fails with a SyntaxError containing `*` where `_` belongs
You pasted the cell from rendered Markdown somewhere, which turns `_name` into
`*name`. Re-upload this `.ipynb` rather than copying cell text between apps.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(REPO)}")
print(f"cells: {len(cells)}  ({sum(1 for c in cells if c.cell_type=='code')} code)")
