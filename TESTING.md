## Manual Verification Steps

### Prerequisites
- Python 3.10+
- Dependencies installed: `pip install h2o pandas shap aiohttp matplotlib`

### Prepare Model
1) Ensure dataset exists:
```
python hands-on/1_generate_dga_data.py
```
2) Train and export MOJO:
```
python 1_train_and_export.py
```
Verify that `model/DGA_Leader.zip` exists.

### Test with a Legitimate Domain
1) Set API key (optional, for playbook):
```
$env:GOOGLE_API_KEY="YOUR_API_KEY"  # PowerShell
```
2) Run analysis:
```
python 2_analyze_domain.py github.com
```
3) Expected outcome:
- DGA probability should be relatively low (e.g., < 0.5)
- Explanation mentions relatively low entropy and moderate length reducing risk
- If API key set, a short 3-5 step playbook is printed

### Test with a DGA-like Domain
1) Run analysis:
```
python 2_analyze_domain.py kq3v9z7j1x5f8g2h.info
```
2) Expected outcome:
- DGA probability should be high (e.g., >= 0.5)
- Explanation cites high entropy and long length increasing risk
- If API key set, a tailored playbook is printed

### Troubleshooting
- If MOJO not found: run `python 1_train_and_export.py` again.
- If Java/H2O issues: ensure internet connectivity for H2O to fetch runtime; retry.
- If Gemini API error: verify `GOOGLE_API_KEY` and account quota.


