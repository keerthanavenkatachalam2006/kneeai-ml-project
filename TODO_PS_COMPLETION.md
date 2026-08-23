# PS Completion — TODO

**Target flow:**
MRI → AI Segmentation → Meniscus → Unified 3D Knee → Measurements → OA Analysis → Implant Ranking → 3D Virtual Fitting → Explainable Result

**Current state:**
- SegResNet trained 20 epochs, continuing to 60 (running now)
- Measurement methodology FIXED (axis bug corrected, joint-line anchoring)
- Case 405: Femur ML 0.0% error, Tibia ML 2.3% error
- Tibial implant: 80% agreement, femoral: 40%
- Full pipeline working end-to-end
- Demo script: `python scripts/run_demo.py --case 405`

---

## Priority 1 — AI Inference Integration

- [x] 1.1 Create `scripts/run_inference.py` — takes a NIfTI path, loads best_model.pth, runs sliding window inference, saves predicted label volume
- [x] 1.2 Verify `KneeSegmentationModel` class in `inference.py` works end-to-end on a test case
- [ ] 1.3 Continue training to 50-100 epochs (Dice still improving at epoch 20, LR was decayed too fast — use warmup + longer cosine schedule)
- [x] 1.4 Validate predictions visually on 3-5 test cases (save mid-slice PNGs with overlay)

## Priority 2 — End-to-End Pipeline (AI → 3D → Measurements)

- [x] 2.1 Create `scripts/run_full_pipeline.py` — single script that does: load MRI → AI segment → measure → reconstruct 3D → match implants → output patient profile
- [x] 2.2 Compare AI-derived measurements vs ground-truth measurements on the same case (sanity check)
- [ ] 2.3 Handle edge cases: what if AI misses a structure entirely (0 voxels)? Add graceful fallbacks in measurement engine

## Priority 3 — Meniscus Segmentation

- [x] 3.1 Research practical options — chose anatomical-prior estimation (geometric approach using adjacent structures)
- [x] 3.2 Implement as `backend/app/segmentation/meniscus_model.py`
- [x] 3.3 Create unified inference that combines bone/cartilage model + meniscus estimation into single label volume (label 6 = Medial Meniscus)
- [x] 3.4 Implement meniscus thickness measurement (anterior/middle/posterior regions)
- [ ] 3.5 Feed meniscus measurements into OA research engine

## Priority 4 — Implant Matching Integration

- [x] 4.1 Wire AI-predicted measurements directly into `ImplantMatcher.match_all()`
- [x] 4.2 Generate ranked implant candidates with full explanations from AI predictions
- [ ] 4.3 Add confidence flag: if AI Dice/confidence is low for a structure, warn that implant sizing may be unreliable
- [x] 4.4 Output structured JSON patient profile (demographics + measurements + OA + implant candidates)

## Priority 5 — 3D Visualization

- [x] 5.1 Create a standalone HTML/JS viewer using Three.js that loads the GLB file
- [x] 5.2 Add toggle controls: show/hide Femur, Tibia, Femoral Cartilage, MTC, LTC, Meniscus
- [x] 5.3 Overlay measurement annotations (measurements displayed in side panel)
- [ ] 5.4 Add implant overlay: load simplified implant geometry, position at tibial plateau / femoral condyle
- [ ] 5.5 Candidate switching: dropdown to swap between Size S / M / L implant overlays
- [ ] 5.6 Color-coded fit visualization: green=optimal, yellow=acceptable, red=overhang risk

## Priority 6 — Validation (Lightweight)

- [x] 6.1 Run AI inference on test cases (5 validated, script supports all 103)
- [x] 6.2 Compute per-class Dice, mean Dice, and per-class volume error vs ground truth
- [x] 6.3 Run measurement engine on both AI predictions and ground truth for same cases
- [x] 6.4 Report measurement error: MAE for femur ML, AP, tibia ML, AP, cartilage thickness
- [x] 6.5 Report implant size agreement: does AI-derived sizing match GT-derived sizing?
- [x] 6.6 Save validation results as `results/validation_report.json`

---

## Blocked / Deferred (DO NOT WORK ON)

- ShapeMed-Knee morphology
- CT integration
- fastMRI self-supervised pretraining
- Frontend React redesign
- Additional statistical dashboards
- Robotics/surgical planning
- Extensive cross-dataset validation beyond SKI10
- Additional audits or metadata work

---

## Execution Order

```
1.1 → 1.2 → 2.1 → 2.2        (get AI pipeline working)
     ↕
1.3 (training continues in background)
     ↓
3.1 → 3.2 → 3.3 → 3.4        (meniscus solved)
     ↓
4.1 → 4.2 → 4.3 → 4.4        (implant matching live)
     ↓
5.1 → 5.2 → 5.3 → 5.4        (3D viewer)
     ↓
6.1 → 6.2 → 6.3 → 6.4 → 6.5 (validation)
     ↓
1.4 + 5.5 + 5.6               (polish)
```

---

## Quick Wins (can do immediately)

1. ~~`run_inference.py` — the model and inference class already exist, just wire them together~~ ✓ DONE
2. ~~`run_full_pipeline.py` — stitch existing modules: inference → measurements → implant matcher → export~~ ✓ DONE
3. Continue training in background (fix LR schedule, run 80 more epochs)

## Current Validation Results (20-epoch model, 5 test cases)

| Structure | Dice |
|---|---|
| Femur | 0.71 |
| Femoral Cartilage | 0.75 |
| Tibia | 0.82 |
| Medial Tibial Cartilage | 0.25 |
| Lateral Tibial Cartilage | 0.32 |
| **Overall Mean** | **0.57** |

Cartilage thickness MAE: FC 0.88mm, MTC 2.27mm, LTC 1.86mm
Tibial implant agreement: 80% (4/5 exact match)
