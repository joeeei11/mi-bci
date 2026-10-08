param([string]$Version = "0.1.0")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$nsisArchive = Join-Path $repoRoot ".tools\nsis-3.0.4.1.7z"
$nsisRoot = Join-Path $repoRoot ".tools\nsis"
$makensis = Join-Path $nsisRoot "makensis.exe"
$expectedHash = "9877DF902530F96357D13A7A31AE2B9DF67F48B11FFC9A1700A7C961574EC5FA"
$cacheRoot = Join-Path $repoRoot ".build-cache"
$previousEnvironment = @{}
foreach ($name in @("UV_CACHE_DIR", "PYINSTALLER_CONFIG_DIR", "TEMP", "TMP", "MPLCONFIGDIR")) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
foreach ($name in @("uv", "pyinstaller", "temp", "matplotlib")) {
    New-Item -ItemType Directory -Path (Join-Path $cacheRoot $name) -Force | Out-Null
}
$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:PYINSTALLER_CONFIG_DIR = Join-Path $cacheRoot "pyinstaller"
$env:TEMP = Join-Path $cacheRoot "temp"
$env:TMP = Join-Path $cacheRoot "temp"
$env:MPLCONFIGDIR = Join-Path $cacheRoot "matplotlib"

Push-Location $repoRoot
try {
    if (-not (Test-Path $venvPython)) {
        uv venv .venv --python 3.11
        if ($LASTEXITCODE -ne 0) { throw "Python environment creation failed" }
    }
    uv pip install --python $venvPython -e '.[desktop,acquisition,research,ai]' 'pyinstaller>=6,<7'
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

    & $venvPython -m PyInstaller --noconfirm --clean --onedir --windowed --name MI-BCI --paths src --add-data ".scratch/mi-bci-research-platform/prototype;prototype" release/mi_bci_desktop.py
    if ($LASTEXITCODE -ne 0) { throw "Desktop build failed" }
    & $venvPython -m PyInstaller --noconfirm --clean --onedir --console --name MI-BCI-MCP --paths src release/mi_bci_mcp.py
    if ($LASTEXITCODE -ne 0) { throw "MCP build failed" }
    $bundledIcu = Join-Path $repoRoot "dist\MI-BCI\_internal\icuuc.dll"
    if (Test-Path $bundledIcu) {
        Remove-Item -LiteralPath $bundledIcu
    }
    $desktopExe = Join-Path $repoRoot "dist\MI-BCI\MI-BCI.exe"
    $smoke = Start-Process -FilePath $desktopExe -ArgumentList "--self-test" -PassThru
    if (-not $smoke.WaitForExit(30000)) {
        Stop-Process -Id $smoke.Id -Force
        throw "Frozen desktop smoke test timed out"
    }
    if ($smoke.ExitCode -ne 0) { throw "Frozen desktop smoke test failed" }

    New-Item -ItemType Directory -Path (Split-Path $nsisArchive), $nsisRoot -Force | Out-Null
    if (-not (Test-Path $nsisArchive)) {
        Invoke-WebRequest -Uri "https://github.com/electron-userland/electron-builder-binaries/releases/download/nsis-3.0.4.1/nsis-3.0.4.1.7z" -OutFile $nsisArchive
    }
    if ((Get-FileHash $nsisArchive -Algorithm SHA256).Hash -ne $expectedHash) {
        throw "NSIS archive checksum mismatch"
    }
    if (-not (Test-Path $makensis)) {
        tar -xf $nsisArchive -C $nsisRoot
        if ($LASTEXITCODE -ne 0) { throw "NSIS extraction failed" }
    }
    Push-Location (Join-Path $repoRoot "release")
    try {
        & $makensis "/DAPP_VERSION=$Version" installer.nsi
        if ($LASTEXITCODE -ne 0) { throw "Installer build failed" }
    } finally {
        Pop-Location
    }
} finally {
    Pop-Location
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], "Process")
    }
}
