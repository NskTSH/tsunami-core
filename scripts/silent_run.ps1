param([string]$RunArgs = "", [string]$ExtraEnv = "")
# Repo root = parent of this script's folder (scripts/). No absolute paths — see NSKTSH_DATA/NSKTSH_RESULTS.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$exe = Join-Path $repoRoot 'MacCormackSimulation\x64\Release\MacCormackSimulation.exe'

# Optional extra env, format "KEY1=VAL1;KEY2=VAL2"
if ($ExtraEnv -ne "") {
    foreach ($pair in $ExtraEnv.Split(';')) {
        if ($pair -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
    }
}

# Suppress all WER/error popups for this process and its children:
#   SEM_FAILCRITICALERRORS = 0x0001
#   SEM_NOGPFAULTERRORBOX  = 0x0002
#   SEM_NOOPENFILEERRORBOX = 0x8000
Add-Type -Name Native -Namespace Win32 -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetErrorMode(uint uMode);
'@
[Win32.Native]::SetErrorMode(0x0001 -bor 0x0002 -bor 0x8000) | Out-Null

$out = "_silent_out.txt"
$err = "_silent_err.txt"
Remove-Item $out, $err -ErrorAction SilentlyContinue

# Default thread cap so simulations don't hog all cores. Overridable via ExtraEnv.
# Quiet hours 08:00–20:00: 4 threads (user works on the desktop).
# After 20:00 and before 08:00: 12 threads (use the box, leave 4 for system).
if (-not $env:OMP_NUM_THREADS) {
    $hr = (Get-Date).Hour
    $defaultThreads = if ($hr -ge 20 -or $hr -lt 8) { 12 } else { 4 }
    $env:OMP_NUM_THREADS = "$defaultThreads"
    Write-Host ("[silent_run] auto OMP_NUM_THREADS=" + $defaultThreads + " (hour=" + $hr + ")")
}

if ($RunArgs -eq "") {
    $proc = Start-Process -FilePath $exe `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
} else {
    $proc = Start-Process -FilePath $exe `
        -ArgumentList $RunArgs `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
}
# Lower priority so other work stays responsive.
try { $proc.PriorityClass = 'BelowNormal' } catch {}
$proc.WaitForExit()
$code = $proc.ExitCode

Write-Host "ExitCode=$code"
Write-Host "--STDOUT--"
if (Test-Path $out) { Get-Content $out -Tail 200 }
Write-Host "--STDERR--"
if (Test-Path $err) { Get-Content $err -Tail 200 }
exit 0
