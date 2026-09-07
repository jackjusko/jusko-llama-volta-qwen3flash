# SPEED-SWEEP harness. One JSONL row per workload. See SPEED-SWEEP.md.
# Usage:
#   .\bench-matrix.ps1 -ConfigId baseline -Workloads curve
#   .\bench-matrix.ps1 -ConfigId layer32k -Restart -ExtraArgs "-sm layer -c 32768" -Workloads promote
param(
  [string]$ConfigId = "baseline",
  [string]$UriBase = "http://127.0.0.1:8082",
  [string]$Results = "H:\Ling-3.0-flash\qwen3.8-flash-next\bench-results.jsonl",
  [string]$Workloads = "curve",
  [string]$ExtraArgs = "",
  [string]$EnvPrefix = "",
  [string]$Nctx = "32768",
  [string]$Split = "tensor",
  [string]$Ubatch = "1024",
  [switch]$Restart,
  [switch]$EraseBeforeThink,
  [string]$LogPath = "",
  [int]$TimeoutSec = 600
)

$ErrorActionPreference = "Stop"
$Root = "H:\Ling-3.0-flash\qwen3.8-flash-next"
if (-not $LogPath) { $LogPath = Join-Path $Root "server-sweep-$ConfigId.log" }
$ChatUri = "$UriBase/v1/chat/completions"
$CompUri = "$UriBase/v1/completions"

function Wait-Server {
  param([int]$Seconds = 80)
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    if ($LogPath -and (Test-Path $LogPath)) {
      $tail = Get-Content $LogPath -Tail 30 -ErrorAction SilentlyContinue | Out-String
      if ($tail -match 'out of memory|failed to create context|failed to allocate compute|GGML_ASSERT') {
        return $false
      }
    }
    try {
      $resp = Invoke-WebRequest -Uri "$UriBase/health" -UseBasicParsing -TimeoutSec 2
      if ($resp.StatusCode -eq 200) { return $true }
    } catch {}
    Start-Sleep -Milliseconds 400
  }
  return $false
}

function Restart-SweepServer {
  param([string]$ArgsTail)
  Write-Host "Restarting llama-server extra='$ArgsTail'"
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Seconds 1
  if ($LogPath -and (Test-Path $LogPath)) { Remove-Item $LogPath -Force -ErrorAction SilentlyContinue }
  $cmd = "cd /d `"$Root`" && set SPEC=0&& $EnvPrefix run-server.cmd $ArgsTail > `"$LogPath`" 2>&1"
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden
  if (-not (Wait-Server 300)) {
    throw "server did not become healthy: $LogPath"
  }
}

function Erase-Slot0 {
  try {
    Invoke-RestMethod -Uri "$UriBase/slots/0?action=erase" -Method Post -TimeoutSec 30 | Out-Null
    Write-Host "erased slot 0"
  } catch {
    Write-Host "slot erase failed: $_"
  }
}

function Get-LlamaWsGb {
  $p = Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($p) { [math]::Round($p.WorkingSet64 / 1GB, 2) } else { 0 }
}

function Get-GpuUtil {
  $raw = nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>$null
  $map = @{}
  foreach ($line in ($raw -split "`r?`n")) {
    if ($line -notmatch ',') { continue }
    $p = $line.Split(',')
    $map[[int]$p[0].Trim()] = [int]$p[1].Trim()
  }
  return $map
}

function Get-DiskH {
  try {
    $c = Get-Counter '\PhysicalDisk(*)\Disk Bytes/sec' -ErrorAction Stop
    $h = $c.CounterSamples | Where-Object { $_.InstanceName -match 'h:' } | Select-Object -First 1
    if ($h) { [int64]$h.CookedValue } else { 0 }
  } catch { 0 }
}

function Get-MetricNTokensMax {
  try {
    $m = (Invoke-WebRequest -Uri "$UriBase/metrics" -UseBasicParsing -TimeoutSec 10).Content
    if ($m -match 'llamacpp:n_tokens_max (\d+)') { [int64]$Matches[1] } else { -1 }
  } catch { -1 }
}

function Invoke-Chat {
  param($Prompt, [int]$MaxTokens, [bool]$Think, [string]$Endpoint = "chat")
  if ($Endpoint -eq "completions") {
    $bodyObj = @{
      model = "qwen3.8-flash-next"
      prompt = $Prompt
      max_tokens = $MaxTokens
      temperature = 0
    }
    $url = $CompUri
  } else {
    $bodyObj = @{
      model = "qwen3.8-flash-next"
      messages = @(@{ role = "user"; content = $Prompt })
      max_tokens = $MaxTokens
      temperature = 0
      chat_template_kwargs = @{ enable_thinking = $Think }
    }
    $url = $ChatUri
  }
  $body = $bodyObj | ConvertTo-Json -Depth 6

  $gpu0 = New-Object System.Collections.Generic.List[int]
  $gpu1 = New-Object System.Collections.Generic.List[int]
  $disk = New-Object System.Collections.Generic.List[int64]
  $job = Start-Job -ScriptBlock {
    param($u, $b, $to)
    Invoke-RestMethod -Uri $u -Method Post -ContentType "application/json" -Body $b -TimeoutSec $to
  } -ArgumentList $url, $body, $TimeoutSec

  while ($job.State -eq "Running") {
    Start-Sleep -Milliseconds 400
    $g = Get-GpuUtil
    if ($g.ContainsKey(0)) { $gpu0.Add($g[0]) }
    if ($g.ContainsKey(1)) { $gpu1.Add($g[1]) }
    $disk.Add((Get-DiskH))
  }

  $r = Receive-Job $job -Wait -ErrorAction Stop
  Remove-Job $job -Force -ErrorAction SilentlyContinue
  $t = $r.timings
  if (-not $t) { throw "no timings in response" }

  function Avg($list) {
    if (-not $list -or $list.Count -eq 0) { return -1 }
    [math]::Round((($list | Measure-Object -Average).Average), 1)
  }
  function Mx($list) {
    if (-not $list -or $list.Count -eq 0) { return -1 }
    ($list | Measure-Object -Maximum).Maximum
  }

  return [ordered]@{
    prompt_n = $t.prompt_n
    prompt_ms = $t.prompt_ms
    prompt_per_second = $t.prompt_per_second
    predicted_n = $t.predicted_n
    predicted_ms = $t.predicted_ms
    predicted_per_second = $t.predicted_per_second
    cache_n = $t.cache_n
    gpu0_avg = (Avg $gpu0)
    gpu0_max = (Mx $gpu0)
    gpu1_avg = (Avg $gpu1)
    gpu1_max = (Mx $gpu1)
    disk_h_avg = [int64](Avg $disk)
  }
}

function Write-Row {
  param($Workload, $Timings)
  $row = [ordered]@{
    ts = (Get-Date).ToString("o")
    config_id = $ConfigId
    n_ctx = $Nctx
    split = $Split
    ubatch = $Ubatch
    extra_args = $ExtraArgs
    workload = $Workload
    llama_ws_gb = (Get-LlamaWsGb)
    n_tokens_max = (Get-MetricNTokensMax)
  }
  foreach ($k in $Timings.Keys) { $row[$k] = $Timings[$k] }
  ($row | ConvertTo-Json -Compress) | Add-Content -Path $Results -Encoding utf8
  $pp = $Timings.prompt_per_second
  $tg = $Timings.predicted_per_second
  Write-Host ("{0,-18} {1,-16} pp={2,7:N1} ({3} tok)  tg={4,7:N1} ({5} tok)  gpu0={6}/{7} gpu1={8}/{9}" -f `
    $ConfigId, $Workload, $pp, $Timings.prompt_n, $tg, $Timings.predicted_n, `
    $Timings.gpu0_avg, $Timings.gpu0_max, $Timings.gpu1_avg, $Timings.gpu1_max)
}

function Pad-Lorem([int]$N) {
  # Instruction last so the model does not EOS on "ping". Count yields >=64 tokens at temp 0.
  (("lorem ") * [Math]::Max(1, $N)) + "Ignore the filler. Count from 1 to 80 as integers separated by spaces. No other text."
}

function Pad-Ctx([int]$TargetTok) {
  # Measured: this chunk is ~11 tokens. Leave a little room for the chat template.
  $chunk = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
  $need = [Math]::Max(8, [int](($TargetTok - 256) / 11.0))
  $sb = New-Object System.Text.StringBuilder ($need * $chunk.Length + 160)
  for ($i = 0; $i -lt $need; $i++) { [void]$sb.Append($chunk) }
  [void]$sb.Append("Ignore the filler. Count from 1 to 80 as integers separated by spaces. No other text.")
  $sb.ToString()
}

function Pad-Unique([int]$N) {
  $sb = New-Object System.Text.StringBuilder
  for ($i = 0; $i -lt $N; $i++) { [void]$sb.Append("blk$i ") }
  [void]$sb.Append("Ignore the filler. Count from 1 to 80 as integers separated by spaces. No other text.")
  $sb.ToString()
}

if ($Restart) {
  Restart-SweepServer -ArgsTail $ExtraArgs
}

if (-not (Wait-Server 15)) { throw "server not healthy at $UriBase" }

Write-Host "Warm..."
try { Invoke-Chat -Prompt "Say hi." -MaxTokens 4 -Think $false | Out-Null } catch { Write-Host "warm failed: $_" }

$run = @{
  "PP-short" = { Write-Row "PP-short" (Invoke-Chat -Prompt "Reply with a single short sentence about GPUs." -MaxTokens 1 -Think $false) }
  "PP-512" = { Write-Row "PP-512" (Invoke-Chat -Prompt (Pad-Lorem 512) -MaxTokens 1 -Think $false) }
  "PP-2k" = { Write-Row "PP-2k" (Invoke-Chat -Prompt (Pad-Lorem 2048) -MaxTokens 1 -Think $false) }
  "PP-8k" = { Write-Row "PP-8k" (Invoke-Chat -Prompt (Pad-Lorem 8192) -MaxTokens 1 -Think $false) }
  "TG-128-empty" = { Write-Row "TG-128-empty" (Invoke-Chat -Prompt "List 40 distinct GPU model names, comma-separated, no extra commentary." -MaxTokens 128 -Think $false) }
  "TG-64-at-256" = { Write-Row "TG-64-at-256" (Invoke-Chat -Prompt (Pad-Lorem 256) -MaxTokens 64 -Think $false) }
  "TG-64-at-512" = { Write-Row "TG-64-at-512" (Invoke-Chat -Prompt (Pad-Lorem 512) -MaxTokens 64 -Think $false) }
  "TG-64-at-1024" = { Write-Row "TG-64-at-1024" (Invoke-Chat -Prompt (Pad-Lorem 1024) -MaxTokens 64 -Think $false) }
  "TG-64-at-2048" = { Write-Row "TG-64-at-2048" (Invoke-Chat -Prompt (Pad-Lorem 2048) -MaxTokens 64 -Think $false) }
  "TG-64-at-4096" = { Write-Row "TG-64-at-4096" (Invoke-Chat -Prompt (Pad-Lorem 4096) -MaxTokens 64 -Think $false) }
  "TG-64-at-8192" = { Write-Row "TG-64-at-8192" (Invoke-Chat -Prompt (Pad-Lorem 8192) -MaxTokens 64 -Think $false) }
  "PP-20k" = { Write-Row "PP-20k" (Invoke-Chat -Prompt (Pad-Ctx 20000) -MaxTokens 1 -Think $false) }
  "PP-60k" = { Write-Row "PP-60k" (Invoke-Chat -Prompt (Pad-Ctx 60000) -MaxTokens 1 -Think $false) }
  "PP-150k" = { Write-Row "PP-150k" (Invoke-Chat -Prompt (Pad-Ctx 150000) -MaxTokens 1 -Think $false) }
  "TG-64-at-20k" = { Write-Row "TG-64-at-20k" (Invoke-Chat -Prompt (Pad-Ctx 20000) -MaxTokens 64 -Think $false) }
  "TG-64-at-60k" = { Write-Row "TG-64-at-60k" (Invoke-Chat -Prompt (Pad-Ctx 60000) -MaxTokens 64 -Think $false) }
  "TG-64-at-150k" = { Write-Row "TG-64-at-150k" (Invoke-Chat -Prompt (Pad-Ctx 150000) -MaxTokens 64 -Think $false) }
  "TG-think-512" = {
    if ($EraseBeforeThink) { Erase-Slot0 }
    Write-Row "TG-think-512" (Invoke-Chat -Prompt "Explain how NVLink helps multi-GPU inference. Be thorough." -MaxTokens 512 -Think $true)
  }
  "TG-unique-2k" = { Write-Row "TG-unique-2k" (Invoke-Chat -Prompt (Pad-Unique 2048) -MaxTokens 64 -Think $false) }
  "TG-rep-2k" = { Write-Row "TG-rep-2k" (Invoke-Chat -Prompt (Pad-Lorem 2048) -MaxTokens 64 -Think $false) }
}

$sets = @{
  curve = @("PP-short","PP-512","PP-2k","TG-128-empty","TG-64-at-256","TG-64-at-512","TG-64-at-1024","TG-64-at-2048","TG-64-at-4096","TG-think-512")
  "curve-tg" = @("TG-64-at-256","TG-64-at-512","TG-64-at-1024","TG-64-at-2048","TG-64-at-4096")
  core = @("PP-2k","TG-128-empty","TG-64-at-2048")
  screen20 = @("PP-2k","TG-64-at-2048","PP-20k","TG-64-at-20k")
  promote = @("PP-2k","TG-128-empty","TG-64-at-2048","TG-think-512")
  agentic = @("PP-2k","TG-64-at-2048","TG-64-at-8192","PP-20k","TG-64-at-20k","TG-think-512")
  "agentic-20k" = @("PP-20k","TG-64-at-20k")
  "agentic-60k" = @("PP-60k","TG-64-at-60k")
  "agentic-150k" = @("PP-150k","TG-64-at-150k")
  ctx150 = @("PP-2k","TG-64-at-2048","TG-think-512","PP-20k","TG-64-at-20k")
  pp = @("PP-short","PP-512","PP-2k","PP-8k")
  think = @("TG-think-512")
  "unique-rep" = @("TG-rep-2k","TG-unique-2k")
  all = @("PP-short","PP-512","PP-2k","PP-8k","TG-128-empty","TG-64-at-256","TG-64-at-512","TG-64-at-1024","TG-64-at-2048","TG-64-at-4096","TG-64-at-8192","TG-think-512","TG-unique-2k")
}

$names = $sets[$Workloads]
if (-not $names) { throw "unknown -Workloads $Workloads" }
$failed = 0
foreach ($n in $names) {
  try { & $run[$n] } catch {
    Write-Host "FAIL $n : $_"
    $failed++
    Write-Row "$n-FAIL" ([ordered]@{
      prompt_n = 0; prompt_ms = 0; prompt_per_second = 0
      predicted_n = 0; predicted_ms = 0; predicted_per_second = 0
      cache_n = 0; gpu0_avg = -1; gpu0_max = -1; gpu1_avg = -1; gpu1_max = -1
      disk_h_avg = 0; error = "$_"
    })
  }
}
if ($LogPath -and (Test-Path $LogPath)) {
  $tail = Get-Content $LogPath -Tail 40 -ErrorAction SilentlyContinue | Out-String
  if ($tail -match 'GGML_ASSERT|out of memory|failed to allocate compute') { $failed++ }
}
if ($failed -gt 0) { exit 1 }
