# One action: opens FastAPI (uvicorn) in a NEW window, then starts the UI here.
# Usage:  cd to repo root (folder with backend/ + src/)  ;  .\dev-all.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"

if (-not (Test-Path (Join-Path $backend "app\main.py"))) {
    Write-Host "Expected app\main.py in: $backend" -ForegroundColor Red
    exit 1
}

Write-Host "Starting FastAPI (uvicorn) in a separate window (leave it open)..." -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory $backend -ArgumentList @(
    "-NoExit",
    "-Command",
    "uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
)

Start-Sleep -Seconds 2

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm not found. Install Node.js LTS from https://nodejs.org/ then run this script again." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path (Join-Path $root ".env"))) {
    Set-Content -Path (Join-Path $root ".env") -Encoding utf8 -Value "VITE_API_BASE_URL=http://127.0.0.1:8000"
}

Set-Location $root
if (-not (Test-Path "node_modules")) {
    Write-Host "First-time install..." -ForegroundColor Cyan
    npm install
}

Write-Host "Starting frontend (open the URL below). Ctrl+C stops only the UI; close the FastAPI window separately." -ForegroundColor Cyan
npm run dev
