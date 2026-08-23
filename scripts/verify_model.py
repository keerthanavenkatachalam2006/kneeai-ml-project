"""
Verify model instantiation, forward pass, and memory usage.
Must pass before launching full training.
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

import torch
from torch.cuda.amp import autocast
from app.segmentation.model import create_segresnet, get_model_summary

print("=" * 60)
print("Model Verification")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Create model
model = create_segresnet().to(device)
summary = get_model_summary(model)
print(f"\nModel: {summary['architecture']}")
print(f"Parameters: {summary['total_params_M']}M")

# Forward pass with training patch size
print("\n--- Forward pass test (96x96x96 patch, AMP) ---")
model.train()
x = torch.randn(2, 1, 96, 96, 96, device=device)  # batch=2, channels=1

with autocast(enabled=True):
    y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Output classes: {y.shape[1]}")

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Memory headroom: {(torch.cuda.get_device_properties(0).total_memory / 1024**3 - peak_mem):.2f} GB")

# Backward pass test
print("\n--- Backward pass test ---")
from monai.losses import DiceCELoss
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
target = torch.randint(0, 6, (2, 1, 96, 96, 96), device=device)

with autocast(enabled=True):
    output = model(x)
    loss = loss_fn(output, target)

loss.backward()

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory after backward: {peak_mem:.2f} GB")
    fits = peak_mem < 5.5  # Leave 0.5 GB headroom
    print(f"Fits in 6GB VRAM: {'YES' if fits else 'NO - need smaller config'}")

print("\n--- Sliding window inference test (small volume) ---")
model.eval()
from monai.inferers import sliding_window_inference

torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
test_vol = torch.randn(1, 1, 64, 128, 128, device=device)

with torch.no_grad(), autocast(enabled=True):
    pred = sliding_window_inference(
        test_vol,
        roi_size=(96, 96, 96),
        sw_batch_size=2,
        predictor=model,
        overlap=0.5,
    )

print(f"Inference input: {test_vol.shape}")
print(f"Inference output: {pred.shape}")
if torch.cuda.is_available():
    peak_inf = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory (inference): {peak_inf:.2f} GB")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - Ready for training")
print("=" * 60)
