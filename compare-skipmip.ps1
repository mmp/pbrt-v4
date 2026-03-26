# Compare pbrt runs with and without --skipmip: full logs on disk + timing/memory summary.
#
# Usage (from repo root):
#   .\compare-skipmip.ps1 -Scene "path\to\scene.pbrt"
#   .\compare-skipmip.ps1 -Scene "path\to\scene.pbrt" -Spp 16
#   .\compare-skipmip.ps1 "scene.pbrt" 16
#   .\compare-skipmip.ps1 "scene.pbrt" 16 -ShowProgress --gpu
#   .\compare-skipmip.ps1 "scene.pbrt" -ShowProgress --gpu   (--gpu is not parsed as Spp; use -Gpu or trailing --gpu)
# Omit -Spp and any leading digits-only tail so pbrt.exe does not get --spp (scene Integrator "integer pixelsamples" applies).
#
# Render progress: stdout is written to the .txt log live while pbrt runs (poll ~8 Hz for console echo).
# Optional -ExtraPbrtArgs is appended for both runs (e.g. --wavefront).
# Pass -Gpu or a trailing --gpu (after other args) to add pbrt's --gpu for both runs.
# By default, repetitive "Rendering:" lines are not echoed (full log files unchanged); use -ShowProgress for all lines.
# SkipMip mip preprocess logs one summary line by default; use -VerboseMipPreprocess for per-texture/per-geometry lines (slow).
# SSIM: compare_ssim.py needs pip install numpy scikit-image pillow (full-res skimage SSIM).

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Scene,

    # Named only (no Position): so trailing tokens like --gpu go to RemainingArguments, not here.
    # Optional positional spp: first RemainingArguments token that is all-digits (>= 1) when -Spp omitted.
    [Parameter(Mandatory = $false)]
    $Spp = $null,

    [string] $PbrtExe = "",
    [string] $LogDir = "",
    [Alias('Extra')]
    [string[]] $ExtraPbrtArgs = @(),

    # If set, every "Rendering:" progress line is printed; otherwise only a short note (full log unchanged).
    [switch] $ShowProgress,

    # If set, SkipMip run adds --verbose-mip-preprocess (large mip analysis log; default is quiet).
    [switch] $VerboseMipPreprocess,

    # If set (or pass trailing --gpu), both runs invoke pbrt with --gpu.
    [switch] $Gpu,

    # Catches e.g. trailing --gpu when not using -ExtraPbrtArgs; other tokens are forwarded to pbrt after --gpu.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PbrtExe {
    param([string] $Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit)) {
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $repoRoot = $PSScriptRoot
    $candidates = @(
        (Join-Path $repoRoot "build-gpu\Release\pbrt.exe"),
        (Join-Path $repoRoot "build-gpu\Debug\pbrt.exe"),
        (Join-Path $repoRoot "build-gpu\pbrt.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return (Resolve-Path -LiteralPath $c).Path }
    }
    throw "Could not find pbrt.exe. Pass -PbrtExe or build under build\Release\pbrt.exe."
}

function Read-LogText([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    return [System.IO.File]::ReadAllText($Path)
}

function Match-One {
    param([string] $Text, [string] $Pattern)
    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $m.Success) { return $null }
    return $m.Groups[1].Value.Trim()
}

function Convert-MemoryToBytes([string] $MemStr) {
    if ([string]::IsNullOrWhiteSpace($MemStr)) { return $null }
    $t = $MemStr.Trim() -replace '\s+', ' '
    if ($t -notmatch '^([\d.]+)\s+(kB|MiB|GiB)$') { return $null }
    $n = [double]$Matches[1]
    switch ($Matches[2]) {
        'kB' { return [long][math]::Round($n * 1024) }
        'MiB' { return [long][math]::Round($n * 1024 * 1024) }
        'GiB' { return [long][math]::Round($n * 1024 * 1024 * 1024) }
    }
    return $null
}

# RAM SkipMip saves vs Full Mip as % of Full Mip (positive = less RAM with SkipMip; negative = SkipMip used more).
function Format-SavingsPercent([int64] $FullMipBytes, [int64] $SkipMipBytes) {
    if ($FullMipBytes -le 0) { return $null }
    $pct = 100.0 * ($FullMipBytes - $SkipMipBytes) / [double]$FullMipBytes
    if ([math]::Abs($pct) -lt 0.05) { return "0.0%" }
    return "{0:N1}%" -f $pct
}

function Split-LineForDisplay([string] $Line, [int] $MaxLen = 96) {
    if ([string]::IsNullOrEmpty($Line)) { return @("") }
    if ($Line.Length -le $MaxLen) { return @($Line) }
    $out = New-Object System.Collections.Generic.List[string]
    $rest = $Line
    while ($rest.Length -gt $MaxLen) {
        $chunk = $MaxLen
        $space = $rest.LastIndexOf(' ', $MaxLen)
        if ($space -gt $MaxLen / 2) { $chunk = $space }
        $out.Add($rest.Substring(0, $chunk).TrimEnd())
        $rest = $rest.Substring($chunk).TrimStart()
    }
    if ($rest.Length -gt 0) { $out.Add($rest) }
    return , $out.ToArray()
}

function Write-PbrtLineToHost([string] $Line, [switch] $ShowProgress) {
    if (-not $ShowProgress -and $Line -match '^\s*Rendering:') {
        return
    }
    if ($Line -match '^\[mip preprocess\] texture') {
        Write-Host ""
    }
    if ($Line -eq '--- stderr ---') {
        Write-Host ""
    }
    if ($Line -match '^\s*Statistics:') {
        Write-Host ""
    }
    if ($Line -match '^\s*Warning:') {
        $parts = Split-LineForDisplay $Line 100
        for ($i = 0; $i -lt $parts.Count; $i++) {
            if ($i -eq 0) { Write-Host $parts[$i] }
            else { Write-Host ("         {0}" -f $parts[$i]) }
        }
        return
    }
    Write-Host $Line
}

function Invoke-PbrtLogged {
    param(
        [string] $PbrtExe,
        [string[]] $Arguments,
        [string] $LogPath,
        [switch] $ShowProgress
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # Stdout -> log file directly so the log grows live (render progress, etc.). Stderr -> temp,
    # then append (avoids mixing two writers on one file). Do not use `& pbrt 2>&1` (stderr
    # becomes ErrorRecord under $ErrorActionPreference Stop).
    $utf8 = New-Object System.Text.UTF8Encoding $false
    if (Test-Path -LiteralPath $LogPath) {
        Remove-Item -LiteralPath $LogPath -Force
    }
    $errTemp = Join-Path ([System.IO.Path]::GetTempPath()) ("pbrt-compare-err-" + [guid]::NewGuid() + ".txt")
    $proc = Start-Process -FilePath $PbrtExe -ArgumentList $Arguments -PassThru -NoNewWindow `
        -RedirectStandardOutput $LogPath -RedirectStandardError $errTemp

    $seenLines = 0
    $nProgress = 0
    while (-not $proc.HasExited) {
        if (Test-Path -LiteralPath $LogPath) {
            try {
                $all = [System.IO.File]::ReadAllLines($LogPath, $utf8)
            } catch {
                $all = @()
            }
            while ($seenLines -lt $all.Length) {
                $line = $all[$seenLines]
                $seenLines++
                if ($line -match '^\s*Rendering:') { $nProgress++ }
                Write-PbrtLineToHost -Line $line -ShowProgress:$ShowProgress
            }
        }
        Start-Sleep -Milliseconds 120
    }
    $null = $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    Start-Sleep -Milliseconds 80
    if (Test-Path -LiteralPath $LogPath) {
        try {
            $all = [System.IO.File]::ReadAllLines($LogPath, $utf8)
        } catch {
            $all = @()
        }
        while ($seenLines -lt $all.Length) {
            $line = $all[$seenLines]
            $seenLines++
            if ($line -match '^\s*Rendering:') { $nProgress++ }
            Write-PbrtLineToHost -Line $line -ShowProgress:$ShowProgress
        }
    }

    try {
        if (Test-Path -LiteralPath $errTemp) {
            $errLines = [System.IO.File]::ReadAllLines($errTemp, $utf8)
            if ($errLines.Count -gt 0) {
                $swErr = New-Object System.IO.StreamWriter($LogPath, $true, $utf8)
                try {
                    $swErr.WriteLine("")
                    $swErr.WriteLine("--- stderr ---")
                    foreach ($el in $errLines) {
                        $swErr.WriteLine($el)
                    }
                } finally {
                    $swErr.Close()
                }
                foreach ($el in $errLines) {
                    Write-PbrtLineToHost -Line $el -ShowProgress:$ShowProgress
                }
            }
        }
    } finally {
        Remove-Item -LiteralPath $errTemp -Force -ErrorAction SilentlyContinue
    }

    $sw.Stop()
    if (-not $ShowProgress -and $nProgress -gt 0) {
        Write-Host ("  ({0} progress lines omitted here; see log file)" -f $nProgress) -ForegroundColor DarkGray
    }
    return @{
        ExitCode       = $exitCode
        ProcessSeconds = $sw.Elapsed.TotalSeconds
    }
}

function Get-SsimCompareResult {
    param(
        [string] $PathA,
        [string] $PathB,
        [string] $RepoRoot
    )
    $py = Join-Path $RepoRoot "compare_ssim.py"
    if (-not (Test-Path -LiteralPath $py)) {
        return @{ Ok = $false; Value = $null; Message = "compare_ssim.py not found next to compare-skipmip.ps1" }
    }
    if (-not (Test-Path -LiteralPath $PathA) -or -not (Test-Path -LiteralPath $PathB)) {
        return @{ Ok = $false; Value = $null; Message = "one or both output images are missing" }
    }
    $usePy = $null
    if (Get-Command python -ErrorAction SilentlyContinue) { $usePy = "python" }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $usePy = "py" }
    else {
        return @{ Ok = $false; Value = $null; Message = "python not on PATH (install: pip install numpy scikit-image pillow)" }
    }
    try {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        if ($usePy -eq "py") {
            $outLines = & py -3 $py $PathA $PathB 2>&1
        } else {
            $outLines = & python $py $PathA $PathB 2>&1
        }
        $ErrorActionPreference = $prevEap
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            $msg = ($outLines | ForEach-Object { "$_" }) -join " "
            return @{ Ok = $false; Value = $null; Message = $msg.Trim() }
        }
        $last = $outLines | Select-Object -Last 1
        $lastStr = "$last".Trim()
        [double]$v = 0.0
        $parsed = [double]::TryParse(
            $lastStr,
            [System.Globalization.NumberStyles]::Any,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$v)
        if (-not $parsed) {
            return @{ Ok = $false; Value = $null; Message = "could not parse SSIM value: $lastStr" }
        }
        return @{ Ok = $true; Value = $v; Message = "" }
    } catch {
        return @{ Ok = $false; Value = $null; Message = $_.Exception.Message }
    }
}

# --- main ---

$PbrtExe = Resolve-PbrtExe -Explicit $PbrtExe

if (-not (Test-Path -LiteralPath $Scene)) {
    throw "Scene file not found: $Scene"
}
$sceneFull = (Resolve-Path -LiteralPath $Scene).Path

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $PSScriptRoot "compare-skipmip-logs"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogDir = (Resolve-Path -LiteralPath $LogDir).Path

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$base = [System.IO.Path]::GetFileNameWithoutExtension($sceneFull)

$imgNoMip = Join-Path $LogDir ("{0}-{1}-nomip.png" -f $base, $stamp)
$imgSkipMip = Join-Path $LogDir ("{0}-{1}-skipmip.png" -f $base, $stamp)
$logNoMip = Join-Path $LogDir ("{0}-{1}-nomip.txt" -f $base, $stamp)
$logSkipMip = Join-Path $LogDir ("{0}-{1}-skipmip.txt" -f $base, $stamp)
$summaryPath = Join-Path $LogDir ("{0}-{1}-comparison.txt" -f $base, $stamp)

$sppForCli = $null
$sppFromNamed = $false
if ($PSBoundParameters.ContainsKey('Spp')) {
    $sppFromNamed = $true
    if ($null -eq $Spp -or "$Spp" -eq '') {
        throw "Spp was specified but is empty; omit -Spp to use the scene file default."
    }
    try {
        $sppForCli = [int]$Spp
    } catch {
        throw "Invalid Spp value (expected positive integer): $Spp"
    }
    if ($sppForCli -lt 1) {
        throw "Spp must be >= 1 when specified (omit -Spp to use the scene file default)."
    }
}

$tailPbrt = New-Object System.Collections.Generic.List[string]
if ($RemainingArguments) {
    foreach ($a in $RemainingArguments) {
        $tailPbrt.Add($a)
    }
}
if (-not $sppFromNamed -and $tailPbrt.Count -gt 0) {
    $head = $tailPbrt[0]
    if ($head -match '^\d+$') {
        $trySpp = [int]$head
        if ($trySpp -ge 1) {
            $sppForCli = $trySpp
            $tailPbrt.RemoveAt(0)
        }
    }
}

$common = @(
    $sceneFull,
    "--stats"
)
if ($null -ne $sppForCli) {
    $common += @("--spp", "$sppForCli")
}
$common += $ExtraPbrtArgs

$gpuWanted = [bool]$Gpu
$finalTail = New-Object System.Collections.Generic.List[string]
foreach ($a in $tailPbrt) {
    if ($a -eq '--gpu') {
        $gpuWanted = $true
    } else {
        $finalTail.Add($a)
    }
}
$tailPbrt = $finalTail
if ($gpuWanted -and ($common -notcontains '--gpu')) {
    $common += '--gpu'
}
if ($tailPbrt.Count -gt 0) {
    $common += [string[]]$tailPbrt.ToArray()
}

$sppLabel = if ($null -ne $sppForCli) { "$sppForCli" } else { "(scene file default)" }

Write-Host "=== pbrt compare-skipmip ===" -ForegroundColor Cyan
Write-Host "pbrt:    $PbrtExe"
Write-Host "scene:   $sceneFull"
Write-Host "spp:     $sppLabel"
Write-Host "log dir: $LogDir"
if ($gpuWanted) {
    Write-Host "gpu:     --gpu (both runs)" -ForegroundColor DarkGray
}
if (-not $ShowProgress) {
    Write-Host "(progress lines hidden; use -ShowProgress to print every Rendering: line)" -ForegroundColor DarkGray
}
if ($VerboseMipPreprocess) {
    Write-Host "(SkipMip run: --verbose-mip-preprocess enabled)" -ForegroundColor DarkGray
}
Write-Host ""

Write-Host "================================================================" -ForegroundColor Yellow
Write-Host " Run 1/2  |  Full Mipmap Chain" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host ""
$argsNoMip = $common + @("--outfile", $imgNoMip)
$runNoMip = Invoke-PbrtLogged -PbrtExe $PbrtExe -Arguments $argsNoMip -LogPath $logNoMip -ShowProgress:$ShowProgress

Write-Host ""
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host " Run 2/2  |  SkipMip" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host ""
$argsSkipMip = $common + @("--skipmip", "--outfile", $imgSkipMip)
if ($VerboseMipPreprocess) {
    $argsSkipMip += "--verbose-mip-preprocess"
}
$runSkipMip = Invoke-PbrtLogged -PbrtExe $PbrtExe -Arguments $argsSkipMip -LogPath $logSkipMip -ShowProgress:$ShowProgress

$textNo = Read-LogText $logNoMip
$textSkip = Read-LogText $logSkipMip

# Stats block (from pbrt --stats): flexible spacing
$patRss = 'RSS\s*\(current\)\s+([\d.]+\s+(?:kB|MiB|GiB))'
$patWall = 'Wall-clock render time\s+([\d.]+)\s*s'
$patMipWall = '\[mip preprocess\] wall time\s+([\d.]+)\s*s'
$patImgTotal = 'Total \(counters\)\s+([\d.]+\s+(?:kB|MiB|GiB))'

$rssNo = Match-One -Text $textNo -Pattern $patRss
$rssSkip = Match-One -Text $textSkip -Pattern $patRss
$wallNo = Match-One -Text $textNo -Pattern $patWall
$wallSkip = Match-One -Text $textSkip -Pattern $patWall
$imgTotNo = Match-One -Text $textNo -Pattern $patImgTotal
$imgTotSkip = Match-One -Text $textSkip -Pattern $patImgTotal
$mipPreSkip = Match-One -Text $textSkip -Pattern $patMipWall

$rssNoB = Convert-MemoryToBytes $rssNo
$rssSkipB = Convert-MemoryToBytes $rssSkip
$imgNoB = Convert-MemoryToBytes $imgTotNo
$imgSkipB = Convert-MemoryToBytes $imgTotSkip

$wallNoD = if ($wallNo) { [double]$wallNo } else { $null }
$wallSkipD = if ($wallSkip) { [double]$wallSkip } else { $null }
$mipPreD = if ($mipPreSkip) { [double]$mipPreSkip } else { $null }

$exitNo = [int]$runNoMip.ExitCode
$exitSk = [int]$runSkipMip.ExitCode

$ssimRes = Get-SsimCompareResult -PathA $imgNoMip -PathB $imgSkipMip -RepoRoot $PSScriptRoot

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("================================")
[void]$sb.AppendLine("PBRT SkipMip Comparison Report")
[void]$sb.AppendLine("================================")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("Scene : $sceneFull")
[void]$sb.AppendLine("SPP   : $sppLabel")
[void]$sb.AppendLine("GPU   : $(if ($gpuWanted) { '--gpu (yes)' } else { '(no)' })")
[void]$sb.AppendLine("Stamp : $stamp")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("-------------------------------------------")
[void]$sb.AppendLine("Outputs")
[void]$sb.AppendLine("-------------------------------------------")
[void]$sb.AppendLine("Logs:")
[void]$sb.AppendLine("  Full Mip : $logNoMip")
[void]$sb.AppendLine("  SkipMip  : $logSkipMip")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("Images:")
[void]$sb.AppendLine("  Full Mip : $imgNoMip")
[void]$sb.AppendLine("  SkipMip  : $imgSkipMip")
[void]$sb.AppendLine("")
if ($exitNo -ne 0 -or $exitSk -ne 0) {
    [void]$sb.AppendLine("-------------------------------------------")
    [void]$sb.AppendLine("Exit Codes")
    [void]$sb.AppendLine("-------------------------------------------")
    [void]$sb.AppendLine("  Full Mip : $exitNo")
    [void]$sb.AppendLine("  SkipMip  : $exitSk")
    [void]$sb.AppendLine("")
}
[void]$sb.AppendLine("-------------------------------------------")
[void]$sb.AppendLine("Timing")
[void]$sb.AppendLine("-------------------------------------------")
[void]$sb.AppendLine("Entire pbrt.exe Run-Time:")
[void]$sb.AppendLine("  Full Mip : {0:N3}s" -f $runNoMip.ProcessSeconds)
$skipProcessLine = "  SkipMip  : {0:N3}s" -f $runSkipMip.ProcessSeconds
if ($null -ne $mipPreD) {
    $skipProcessLine += "  (Mipmap Preprocess Time: {0:N3} s)" -f $mipPreD
}
[void]$sb.AppendLine($skipProcessLine)
[void]$sb.AppendLine("")
if ($null -ne $wallNoD -and $null -ne $wallSkipD) {
    [void]$sb.AppendLine("Render Time:")
    [void]$sb.AppendLine("  Full Mip : {0:N3} s" -f $wallNoD)
    [void]$sb.AppendLine("  SkipMip  : {0:N3} s" -f $wallSkipD)
    [void]$sb.AppendLine("  Delta    : {0:N3} s" -f ($wallSkipD - $wallNoD))
    [void]$sb.AppendLine("")
}
[void]$sb.AppendLine("-------------------------------------------")
[void]$sb.AppendLine("Memory")
[void]$sb.AppendLine("-------------------------------------------")
[void]$sb.AppendLine("RSS")
[void]$sb.AppendLine("  Full Mip : $(if ($rssNo) { $rssNo } else { 'n/a' })")
[void]$sb.AppendLine("  SkipMip  : $(if ($rssSkip) { $rssSkip } else { 'n/a' })")
if ($null -ne $rssNoB -and $null -ne $rssSkipB) {
    $pctRss = Format-SavingsPercent -FullMipBytes $rssNoB -SkipMipBytes $rssSkipB
    [void]$sb.AppendLine("  Savings  : $pctRss")
}
[void]$sb.AppendLine("")
[void]$sb.AppendLine("Image Textures")
[void]$sb.AppendLine("  Full Mip : $(if ($imgTotNo) { $imgTotNo } else { 'n/a' })")
[void]$sb.AppendLine("  SkipMip  : $(if ($imgTotSkip) { $imgTotSkip } else { 'n/a' })")
if ($null -ne $imgNoB -and $null -ne $imgSkipB) {
    $pctImg = Format-SavingsPercent -FullMipBytes $imgNoB -SkipMipBytes $imgSkipB
    [void]$sb.AppendLine("  Savings  : $pctImg")
}
[void]$sb.AppendLine("")
if ($ssimRes.Ok) {
    $ssimMark = if ($ssimRes.Value -gt 0.99) { [char]0x2713 } else { [char]0x2717 }
    $ssimValStr = ([double]$ssimRes.Value).ToString(
        "N6", [System.Globalization.CultureInfo]::InvariantCulture)
    # Avoid -f with multiple placeholders (can throw FormatError with some hosts/encodings).
    [void]$sb.AppendLine("-------------------------------------------")
    [void]$sb.AppendLine("SSIM (Full Mip vs SkipMip): $ssimValStr $ssimMark")
    [void]$sb.AppendLine("-------------------------------------------")

} else {
    $why = if ($ssimRes.Message) {
        ($ssimRes.Message -replace "[\r\n]+", " ").Trim()
    } else {
        "unknown"
    }
    if ($why.Length -gt 160) { $why = $why.Substring(0, 157) + "..." }
    [void]$sb.AppendLine("-------------------------------------------")
    [void]$sb.AppendLine("SSIM (Full Mip vs SkipMip): n/a - $why")
    [void]$sb.AppendLine("-------------------------------------------")

}

$summary = $sb.ToString()
[System.IO.File]::WriteAllText($summaryPath, $summary, (New-Object System.Text.UTF8Encoding $false))

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Summary  |  written to:" -ForegroundColor Cyan
Write-Host " $summaryPath" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host $summary

if ($exitNo -ne 0 -or $exitSk -ne 0) {
    exit [Math]::Max([Math]::Max($exitNo, $exitSk), 1)
}
exit 0
