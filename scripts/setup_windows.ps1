# Instalacion en Windows (CPU, o GPU NVIDIA si se detecta nvidia-smi).
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) { $conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe" }
if (-not (Test-Path $conda)) { throw "No se encontro conda. Instala Miniconda." }

# Hace que 'conda activate' funcione en nuevas terminales de PowerShell
& $conda init powershell | Out-Null

$envs = & $conda env list
if ($envs -notmatch "tt-a048") {
    # environment.yml usa solo conda-forge (los canales defaults exigen aceptar ToS)
    & $conda env create -f environment.yml
}
$envPath = ((& $conda env list) | Select-String "tt-a048").ToString().Split(" ", [StringSplitOptions]::RemoveEmptyEntries)[-1]
$py = Join-Path $envPath "python.exe"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "GPU NVIDIA detectada: torch con CUDA 12.4"
    & $py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "Sin GPU NVIDIA: torch CPU"
    & $py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -r requirements.txt
& $py -m pip install --no-deps -e .\ml-fastvlm
& $py download_models.py
& $py -m pipeline.hardware
Write-Host "`nListo. Abre una terminal nueva y ejecuta:`n  conda activate tt-a048`n  python app.py"
