# Activate venv and run repro_rag_tests.py with forwarded args
param(
    [string]$Mode = 'inprocess',
    [string]$BaseUrl = 'http://127.0.0.1:8001',
    [string]$Output = 'repro_rag_tests_output.json'
)

Write-Host "Activating virtualenv and running tests (mode=$Mode)"
& .\.venv\Scripts\Activate.ps1
python repro_rag_tests.py --mode $Mode --base-url $BaseUrl --output $Output
$LASTEXITCODE | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Tests passed. Output: $Output" -ForegroundColor Green
