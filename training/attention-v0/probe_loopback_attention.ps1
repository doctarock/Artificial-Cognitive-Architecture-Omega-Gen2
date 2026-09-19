param([int]$TimeoutSec = 10)

$ErrorActionPreference = 'Stop'
$endpointLine = Get-Content -LiteralPath '.env' | Where-Object { $_ -match '^OMEGA_ATTENTION_MODEL_BASE_URL\s*=' } | Select-Object -First 1
$modelLine = Get-Content -LiteralPath '.env' | Where-Object { $_ -match '^OMEGA_ATTENTION_MODEL\s*=' } | Select-Object -First 1
if (-not $endpointLine -or -not $modelLine) { throw 'configured attention endpoint/model missing' }
$endpoint = (($endpointLine -split '=', 2)[1]).Trim('"', "'", ' ')
$model = (($modelLine -split '=', 2)[1]).Trim('"', "'", ' ')
$uri = [uri]$endpoint
if ($uri.Scheme -ne 'http' -or $uri.Host -notin @('127.0.0.1', 'localhost', '::1')) {
    throw 'probe refuses non-loopback model traffic'
}
$url = "$($uri.GetLeftPart([System.UriPartial]::Authority).TrimEnd('/'))/api/generate"
$a = '11111111-1111-4111-8111-111111111111'
$b = '22222222-2222-4222-8222-222222222222'
function Candidate([string]$Id, [double]$Score, [double]$Confidence, [bool]$Resident) {
    return @{id=$Id;kind='observation';activation_total=$Score;surprise=$null;confidence=$Confidence;goal_priority=$null;in_working_memory=$Resident;broadcast_count=0;age_ms=100}
}
$cases = @(
    @{name='ignore';focus=$null;candidates=@((Candidate $a -1.0 0.9 $false));expectedOp='IGNORE';expectedTarget=$null},
    @{name='attend';focus=$null;candidates=@((Candidate $a 2.0 0.9 $false));expectedOp='ATTEND';expectedTarget=$a},
    @{name='maintain';focus=$a;candidates=@((Candidate $a 2.0 0.9 $true),(Candidate $b 0.8 0.9 $false));expectedOp='MAINTAIN';expectedTarget=$a},
    @{name='switch';focus=$a;candidates=@((Candidate $a 0.8 0.9 $true),(Candidate $b 2.0 0.9 $false));expectedOp='SWITCH';expectedTarget=$b},
    @{name='suppress';focus=$a;candidates=@((Candidate $a 2.0 0.2 $true));expectedOp='SUPPRESS';expectedTarget=$a}
)
$times = @()
$passes = 0
foreach ($case in $cases) {
    $workspace = @{protocol='omega-attention-workspace/v0.2';attention_threshold=0.5;working_memory_capacity=4;current_focus=$case.focus;candidates=$case.candidates} | ConvertTo-Json -Depth 8 -Compress
    $request = @{model=$model;prompt="Select the next attention operation for this cognitive workspace:`n$workspace";stream=$false;options=@{temperature=0.0}} | ConvertTo-Json -Depth 8 -Compress
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-RestMethod -Uri $url -Method Post -ContentType 'application/json' -Body $request -TimeoutSec $TimeoutSec
        $parsed = $response.response | ConvertFrom-Json
        $wellFormed = (@('ATTEND','MAINTAIN','SWITCH','SUPPRESS','IGNORE') -contains $parsed.operation) -and
            ($null -eq $parsed.target -or @($case.candidates.id) -contains $parsed.target) -and
            ($parsed.confidence -ge 0 -and $parsed.confidence -le 1) -and
            (-not [string]::IsNullOrWhiteSpace($parsed.reason_code))
        $correct = $wellFormed -and $parsed.operation -eq $case.expectedOp -and $parsed.target -eq $case.expectedTarget
        if ($correct) { $passes++ }
        Write-Output "case=$($case.name) elapsed_ms=$([math]::Round($clock.Elapsed.TotalMilliseconds)) well_formed=$wellFormed correct=$correct operation=$($parsed.operation) confidence=$($parsed.confidence)"
    } catch {
        Write-Output "case=$($case.name) elapsed_ms=$([math]::Round($clock.Elapsed.TotalMilliseconds)) error=$($_.Exception.GetType().Name)"
    } finally {
        $clock.Stop()
        $times += $clock.Elapsed.TotalMilliseconds
    }
}
$sorted = @($times | Sort-Object)
Write-Output "loopback_attention_passes=$passes/$($cases.Count) p50_ms=$([math]::Round($sorted[[math]::Floor($sorted.Count / 2)])) max_ms=$([math]::Round($sorted[-1]))"
