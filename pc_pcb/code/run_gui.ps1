$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notmatch 'MinGW|msys32|msys64|mingw' }) -join ';'
Write-Host "[PATH cleaned] Launching PCB Defect Detection GUI..."
python MainProgram.py
if ($LASTEXITCODE -ne 0) { Read-Host "Press Enter to exit" }
