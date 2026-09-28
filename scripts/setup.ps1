# One-command setup for Windows. Run from the repo root:  powershell -File scripts/setup.ps1
# Needs: git, uv (winget install astral-sh.uv). uv downloads Python 3.12 itself.
$ErrorActionPreference = "Stop"
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
$env:OFFLINE = "1"
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m scripts.check_controls
Write-Host "`nSetup OK. Start the app:  `$env:OFFLINE='1'; .venv\Scripts\uvicorn src.api.main:app --reload  -> http://127.0.0.1:8000/"
