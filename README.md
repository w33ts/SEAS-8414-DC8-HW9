## The Prescriptive DGA Detector

End-to-end tool that:
- Trains a DGA classifier with H2O AutoML
- Explains predictions with SHAP
- Generates a prescriptive, context-aware playbook with Google Gemini

### Repository Layout
- `1_train_and_export.py`: Train with AutoML and export `model/DGA_Leader.zip`
- `2_analyze_domain.py`: Analyze a domain, explain via SHAP, generate a playbook
- `hands-on/`: Original lecture scripts and generated assets
  - `hands-on/1_generate_dga_data.py`: Creates `hands-on/dga_dataset_train.csv`
  - `hands-on/2_run_automl.py`, `hands-on/3_explain_model.py`, `hands-on/4_generate_prescriptive_playbook.py`
  - `hands-on/models/`: Saved binary models; MOJOs stored under `model/`

### Setup
1) Python 3.10+
2) Install dependencies (example):
```
pip install h2o pandas shap aiohttp matplotlib
```
On first import, H2O will download a compatible Java runtime automatically if needed.

### Training
1) Generate training data (optional if already present):
```
python hands-on/1_generate_dga_data.py
```
2) Train and export MOJO:
```
python 1_train_and_export.py
```
Outputs:
- `model/DGA_Leader.zip` (MOJO for scoring)
- `hands-on/models/best_dga_model` (binary model for local checks)

### Analyze a Domain
Set your Google API key (for playbook generation):
- PowerShell (Windows):
```
$env:GOOGLE_API_KEY="YOUR_API_KEY"
```
- macOS/Linux:
```
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Run analysis:
```
python 2_analyze_domain.py suspiciousdomain123.com
```
You will see:
- Prediction and probability
- SHAP explanation summary
- AI-generated prescriptive playbook (if `GOOGLE_API_KEY` is set)

### Notes
- MOJO scoring is used for stable production-like inference.
- SHAP uses KernelExplainer with a small background set for fast local explanations.
- If no API key is set, the script prints the explanation summary and prediction only.


