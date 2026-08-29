param(
    [string]$Checkout = "external/S4M"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptRoot "../..")).Path
$CheckoutPath = Join-Path $ProjectRoot $Checkout
$MetadataPath = Join-Path $ScriptRoot "upstream.json"
$PatchPath = Join-Path $ScriptRoot "patches/s4m-malr-fair.patch"
$OverlayPath = Join-Path $ScriptRoot "overlay/malr_fair"
$Verifier = Join-Path $ScriptRoot "verify_upstream.py"
$ExpectedCommit = "a718823addd3606e763dfc261174e0135b2535f4"

if (-not (Test-Path -LiteralPath (Join-Path $CheckoutPath ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CheckoutPath) | Out-Null
    git clone https://github.com/WINTERWEEL/S4M.git $CheckoutPath
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    git -C $CheckoutPath checkout --detach $ExpectedCommit
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
}

$AlreadyPatched = $false
if (Test-Path -LiteralPath $PatchPath) {
    git -C $CheckoutPath apply --reverse --check $PatchPath 2>$null
    $AlreadyPatched = ($LASTEXITCODE -eq 0)
}

if ($AlreadyPatched) {
    & python $Verifier --root $CheckoutPath --metadata $MetadataPath --allow-patched
    if ($LASTEXITCODE -ne 0) { throw "existing patched checkout failed verification" }
    Write-Host "Official S4M checkout is already patched and verified."
    exit 0
}

& python $Verifier --root $CheckoutPath --metadata $MetadataPath
if ($LASTEXITCODE -ne 0) { throw "fresh checkout failed verification" }
if (-not (Test-Path -LiteralPath $OverlayPath)) {
    throw "overlay is missing: $OverlayPath"
}
if (-not (Test-Path -LiteralPath $PatchPath)) {
    throw "patch is missing: $PatchPath"
}

Copy-Item -LiteralPath $OverlayPath -Destination (Join-Path $CheckoutPath "s4m") -Recurse -Force
git -C $CheckoutPath apply --check $PatchPath
if ($LASTEXITCODE -ne 0) { throw "S4M patch does not apply cleanly" }
git -C $CheckoutPath apply $PatchPath
if ($LASTEXITCODE -ne 0) { throw "S4M patch failed" }

& python $Verifier --root $CheckoutPath --metadata $MetadataPath --allow-patched
if ($LASTEXITCODE -ne 0) { throw "patched checkout failed verification" }
Write-Host "Official S4M checkout prepared at $CheckoutPath"
