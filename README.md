# ECG R‑Peak Detection with 1D CNN (EECS4643)

This repository contains a Jupyter Notebook that preprocesses ECG signals, creates labels for R‑peaks, trains a compact 1D CNN to perform per‑sample peak detection, and applies the trained model to new data. The end-to-end workflow runs inside `EECS4643Project_real.ipynb`.

- Notebook: `EECS4643Project_real.ipynb`
- Language: Python 3
- Frameworks/Libraries: NumPy, SciPy, Matplotlib, scikit‑learn, PyTorch, Requests

## What this project does
1. Load public demo ECG data (1000 Hz) from NeuroKit and a local Biopac file for inference.
2. Preprocess ECG: resample to 1000 Hz, band‑pass filter around the QRS band (~5–15 Hz), and z‑score normalize.
3. Create training labels by detecting R‑peaks on the normalized ECG using a minimum RR interval (refractory period).
4. Train a 1D CNN that outputs a logit per time sample indicating “peak” vs “non‑peak”.
5. Evaluate with per‑sample accuracy, confusion matrix, and training curves.
6. Run inference on new ECG, convert model probabilities to peaks using `find_peaks`, and visualize detections.

## Methods in brief
- Preprocessing
  - Resampling: `scipy.signal.resample_poly` to 1000 Hz.
  - Filtering: zero‑phase Butterworth band‑pass via `butter` + `filtfilt` (≈5–15 Hz) to emphasize QRS energy.
  - Normalization: per‑segment z‑score (mean 0, unit variance).
- Labeling
  - `scipy.signal.find_peaks` on the normalized ECG with a minimum RR interval (e.g., 0.2–0.6 s) to avoid double counting.
- Model (PyTorch)
  - Conv1d(1→32, k=7, pad=3) → ReLU → Conv1d(32→64, k=7, pad=3) → ReLU → Dropout(0.1) → Conv1d(64→1, k=1).
  - Input/Output shape: (B, C, T) → (B, 1, T); one logit per time step.
- Training
  - Loss: `BCEWithLogitsLoss` with large `pos_weight` (e.g., 999) to counter class imbalance (few peaks vs many non‑peaks).
  - Optimizer: SGD (lr=0.01); Epochs ≈ 300; chronological split 80/20 (first 80% train, last 20% validation).
- Inference
  - Apply same preprocessing; run model → sigmoid → per‑sample probabilities; use `find_peaks` with RR‑interval and probability threshold to produce discrete peak indices.

## Getting started
### 1) Set up Python environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install requests numpy matplotlib scipy scikit-learn torch jupyter
```

### 2) Open and run the notebook
- In VS Code: open the folder, open `EECS4643Project_real.ipynb`, select a Python kernel, and Run All cells in order (Cells 1 → end).
- Or with Jupyter:
```powershell
jupyter lab
# or
jupyter notebook
```

Internet access is required in Step 1 to download the NeuroKit demo ECG CSV.

## Where to put your data
- Public demo: auto‑downloaded from NeuroKit.
- Local Biopac example: place the text file (e.g., `Lab-03-L05_ayesha.txt`) in the project root. The notebook loads it with `readBiopacResults('./Lab-03-L05_ayesha.txt')` assuming channel 0 and `sample_rate = 800` Hz.

## Results and expected outputs
- Figures comparing raw vs preprocessed ECG (first ~30 s).
- Training curves (accuracy and loss for train/val).
- Validation summary: accuracy, confusion matrix, classification report.
- Inference plot: resampled ECG with red markers at detected peaks.

## Troubleshooting
- PyTorch dtype mismatch: `RuntimeError: Input type (double) and bias type (float) should be the same`
  - Cause: NumPy defaults to float64 (double) while PyTorch layers default to float32 (float).
  - Fix: ensure tensors are float32 when created, e.g. `torch.tensor(arr, dtype=torch.float32)`; also make targets float32 for `BCEWithLogitsLoss`.
- No peaks detected or too many peaks
  - Tune `min_rr_sec` (physiological refractory period) and/or the probability threshold used in `find_peaks` during inference.
- Poor validation accuracy
  - Revisit preprocessing (filter band), increase epochs, adjust `pos_weight`, or consider a slightly deeper CNN.

## References (GitHub)
- NumPy: https://github.com/numpy/numpy
- SciPy (signal): https://github.com/scipy/scipy
  - `find_peaks`, `butter`, `filtfilt`, `resample_poly`
- Matplotlib: https://github.com/matplotlib/matplotlib
- scikit‑learn (metrics): https://github.com/scikit-learn/scikit-learn
- PyTorch: https://github.com/pytorch/pytorch
  - `Conv1d`, `BCEWithLogitsLoss`, training utilities
- NeuroKit2 (data source repo): https://github.com/neuropsychology/NeuroKit

Additional ECG toolkits and inspiration
- BioSPPy: https://github.com/PIA-Group/BioSPPy
- HeartPy: https://github.com/paulvangentcom/heartrate_analysis_python
- py‑ecg‑detectors: https://github.com/berndporr/py-ecg-detectors
- WFDB Python (PhysioNet I/O): https://github.com/MIT-LCP/wfdb-python
- tsai (time‑series DL): https://github.com/timeseriesAI/tsai
- Temporal Convolutional Networks: https://github.com/locuslab/TCN
- PyTorch examples: https://github.com/pytorch/examples

## Notes
- The notebook assumes a single‑channel ECG input and preserves sequence length through padding.
- All inference signals are resampled to the training rate (1000 Hz) before model application.
- Peak picking is performed on the model’s probability trace, enforcing a minimum RR interval to avoid double detections.
