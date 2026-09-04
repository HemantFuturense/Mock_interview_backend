$ErrorActionPreference = 'Stop'

$backendRoot = $PSScriptRoot
$workspaceRoot = (Resolve-Path (Join-Path $backendRoot '..\..\..')).Path
$python = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}

Set-Location -LiteralPath $backendRoot
& $python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
