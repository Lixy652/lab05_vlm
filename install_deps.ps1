$ErrorActionPreference = 'Stop'
$python = $env:HW5_PYTHON
if (-not $python -or -not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}
$target = Join-Path $PSScriptRoot '.deps'
& $python -m pip install --disable-pip-version-check --no-cache-dir --target $target `
    --index-url https://download.pytorch.org/whl/cu124 `
    --extra-index-url https://pypi.org/simple `
    -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
