<#
.SYNOPSIS
  Windows/PowerShell task runner for MORP. The Makefile equivalent.

.DESCRIPTION
  The Makefile in this repository is POSIX-only: it needs `make`, and its paths assume a
  Unix virtualenv layout (.venv/bin rather than .venv\Scripts). On Windows use this script
  instead - same target names.

      .\morp.ps1 help
      .\morp.ps1 site-install
      .\morp.ps1 site

  The leading .\ is required. PowerShell does not search the current directory for
  commands, which is why `make site-install` failed rather than falling back to anything.

  If PowerShell refuses to run the file ("running scripts is disabled on this system"),
  allow it for this session only:

      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

  Every target prints the command it runs, so anything here can also be run by hand.

  Kept deliberately ASCII-only: Windows PowerShell 5.1 decodes a .ps1 file with the system
  code page unless it carries a BOM, so a UTF-8 arrow or em dash in this file would come
  out as mojibake in the console.
#>

[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string]$Target = 'help',

  # For the `infer` target, e.g. -Case oaizib-405
  [string]$Case
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

# --- helpers ---------------------------------------------------------------------------

function Write-Step([string]$text) {
  Write-Host "> $text" -ForegroundColor DarkCyan
}

function Assert-Tool([string]$name, [string]$hint) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "'$name' was not found on PATH. $hint"
  }
}

# No activated virtualenv is assumed: a local .venv is used when it exists, otherwise the
# interpreter on PATH.
function Get-Python {
  $venv = Join-Path $Root '.venv\Scripts\python.exe'
  if (Test-Path $venv) { return $venv }
  Assert-Tool 'python' 'Install Python 3.11+ and reopen the terminal.'
  return 'python'
}

<#
  Run npm in one of the front-end folders.

  npm.cmd is named explicitly rather than relying on `npm`: Node ships npm.cmd, npm.ps1 and
  a bare npm shim, and which one PowerShell picks changes how arguments are passed. The
  array is passed without the splat operator, which PowerShell expands into separate
  arguments for a native command.
#>
function Invoke-Npm([string]$relative, [string[]]$npmArgs) {
  Assert-Tool 'npm' 'Install Node.js 20+ from https://nodejs.org and reopen the terminal.'

  $dir = Join-Path $Root $relative
  if (-not (Test-Path $dir)) { throw "Directory not found: $dir" }

  Push-Location $dir
  try {
    Write-Step ("npm " + ($npmArgs -join ' ') + "   (in $relative)")
    & npm.cmd $npmArgs
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "npm exited with $code" }
  }
  finally {
    Pop-Location
  }
}

function Show-Help {
  $rows = @(
    @('help', 'This list'),
    @('', ''),
    @('install', 'Python package (imaging + api + dev extras)'),
    @('web-install', 'Install workspace dependencies (apps/web)'),
    @('site-install', 'Install showcase site dependencies (site)'),
    @('install-all', 'All three of the above, in order'),
    @('', ''),
    @('api', 'FastAPI backend on 127.0.0.1:8000 (blocks)'),
    @('web', 'Surgeon workspace on localhost:3000 (blocks)'),
    @('site', 'Showcase site on localhost:3001 (blocks)'),
    @('', ''),
    @('site-build', 'Static export of the showcase site to site/out'),
    @('site-shots', 'Retake showcase screenshots (needs api + web running)'),
    @('web-build', 'Production build of the workspace'),
    @('web-clean', 'Delete apps/web/.next'),
    @('', ''),
    @('typecheck', 'tsc on both front ends'),
    @('test', 'pytest'),
    @('lint', 'ruff check'),
    @('demo', 'End-to-end pipeline on the synthetic phantom knee'),
    @('infer', 'Segment + plan one case:  .\morp.ps1 infer -Case oaizib-405')
  )

  Write-Host ''
  Write-Host '  MORP - task runner for Windows' -ForegroundColor White
  Write-Host '  usage: .\morp.ps1 <target>' -ForegroundColor DarkGray
  Write-Host ''
  foreach ($row in $rows) {
    if ($row[0] -eq '') {
      Write-Host ''
    }
    else {
      Write-Host ('  {0,-14} ' -f $row[0]) -ForegroundColor Cyan -NoNewline
      Write-Host $row[1] -ForegroundColor Gray
    }
  }
  Write-Host ''
  Write-Host '  Full stack = three terminals: api, web, site.' -ForegroundColor DarkGray
  Write-Host ''
}

# --- targets ---------------------------------------------------------------------------

switch ($Target.ToLowerInvariant()) {

  'help' { Show-Help }

  'install' {
    $py = Get-Python
    Write-Step "$py -m pip install -e `".[imaging,api,dev]`""
    & $py -m pip install -e ".[imaging,api,dev]"
  }

  'web-install' { Invoke-Npm 'apps\web' @('install') }

  'site-install' { Invoke-Npm 'site' @('install') }

  'install-all' {
    & $PSCommandPath 'install'
    & $PSCommandPath 'web-install'
    & $PSCommandPath 'site-install'
  }

  'api' {
    $py = Get-Python
    Write-Step 'uvicorn kneeai.api.main:app --reload --host 127.0.0.1 --port 8000'
    Write-Host '  Loopback only: this API has no authentication.' -ForegroundColor DarkYellow
    & $py -m uvicorn kneeai.api.main:app --reload --host 127.0.0.1 --port 8000
  }

  'web' { Invoke-Npm 'apps\web' @('run', 'dev') }

  'site' { Invoke-Npm 'site' @('run', 'dev') }

  'web-build' {
    Write-Host '  Stop the web dev server first: both write to apps\web\.next.' -ForegroundColor DarkYellow
    Invoke-Npm 'apps\web' @('run', 'build')
  }

  'web-clean' {
    $next = Join-Path $Root 'apps\web\.next'
    if (Test-Path $next) {
      Write-Step "remove $next"
      Remove-Item -Recurse -Force $next
    }
    Write-Host 'cleared apps\web\.next - restart the web dev server' -ForegroundColor Gray
  }

  'site-build' { Invoke-Npm 'site' @('run', 'build') }

  'site-shots' {
    Write-Host '  Needs the API (8000) and the workspace (3000) already running.' -ForegroundColor DarkYellow
    Invoke-Npm 'site' @('run', 'shots')
  }

  'typecheck' {
    Invoke-Npm 'apps\web' @('run', 'typecheck')
    Invoke-Npm 'site' @('run', 'typecheck')
  }

  'test' {
    $py = Get-Python
    Write-Step "$py -m pytest"
    & $py -m pytest
  }

  'lint' {
    $py = Get-Python
    Write-Step "$py -m ruff check src tests"
    & $py -m ruff check src tests
  }

  'demo' {
    $py = Get-Python
    Write-Step 'kneeai demo run --out artifacts/demo'
    & $py -m kneeai demo run --out artifacts/demo
  }

  'infer' {
    if (-not $Case) { throw 'Pass a case, e.g. .\morp.ps1 infer -Case oaizib-405' }
    $py = Get-Python
    $file = ($Case -replace '^oaizib-', 'oaizib_')
    $image = "data/raw/oaizib/imagesTs/${file}_0000.nii.gz"
    Write-Step "kneeai infer --case-id $Case --image $image"
    & $py -m kneeai infer --case-id $Case --source ai_prediction --image $image --side right --strict
  }

  default {
    Write-Host "Unknown target '$Target'." -ForegroundColor Red
    Show-Help
    exit 1
  }
}
