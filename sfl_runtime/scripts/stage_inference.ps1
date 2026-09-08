param(
    [Parameter(Mandatory)][string]$DecoderPte,
    [Parameter(Mandatory)][string]$Tokenizer,
    [Parameter(Mandatory)][string]$EncoderPte,
    [Parameter(Mandatory)][string]$EmbeddingDirectory,
    [Parameter(Mandatory)][string]$AlignmentCheckpoint,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [int]$EncoderInputLength = 240,
    [int]$EncoderOutputWidth = 768,
    [int]$SensorTokens = 162,
    [int]$MaxNewTokens = 96
)

$ErrorActionPreference = 'Stop'
$inputs = @($DecoderPte, $Tokenizer, $EncoderPte, $EmbeddingDirectory, $AlignmentCheckpoint)
foreach ($inputPath in $inputs) {
    if (-not (Test-Path -LiteralPath $inputPath)) { throw "Inference asset does not exist: $inputPath" }
}
if ($EncoderInputLength -le 0 -or $EncoderOutputWidth -le 0 -or
    $SensorTokens -le 0 -or $MaxNewTokens -le 0) {
    throw 'Inference dimensions and max-new-token count must be positive'
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) {
    if (Get-ChildItem -LiteralPath $output -Force | Select-Object -First 1) {
        throw "Output directory must be empty: $output"
    }
} else {
    New-Item -ItemType Directory -Path $output | Out-Null
}
$assets = Join-Path $output 'assets'
New-Item -ItemType Directory -Path $assets | Out-Null
Copy-Item -LiteralPath $DecoderPte -Destination (Join-Path $assets 'llama-decoder.pte')
Copy-Item -LiteralPath $Tokenizer -Destination (Join-Path $assets 'tokenizer.json')
Copy-Item -LiteralPath $EncoderPte -Destination (Join-Path $assets 'sensor-encoder.pte')
Copy-Item -LiteralPath $EmbeddingDirectory -Destination (Join-Path $assets 'llama-embedding') -Recurse
Copy-Item -LiteralPath $AlignmentCheckpoint -Destination (Join-Path $assets 'alignment') -Recurse

$deployment = [ordered]@{
    model_pte = 'assets/llama-decoder.pte'
    tokenizer = 'assets/tokenizer.json'
    encoder_pte = 'assets/sensor-encoder.pte'
    embedding_dir = 'assets/llama-embedding'
    alignment_checkpoint = 'assets/alignment'
    decoder_method = 'forward'
    encoder_input_length = $EncoderInputLength
    encoder_output_width = $EncoderOutputWidth
    sensor_tokens = $SensorTokens
    max_new_tokens = $MaxNewTokens
    channels = @(
        [ordered]@{kind=1; value_index=0; mean=0.0; std=1.0},
        [ordered]@{kind=1; value_index=1; mean=0.0; std=1.0},
        [ordered]@{kind=1; value_index=2; mean=0.0; std=1.0},
        [ordered]@{kind=2; value_index=0; mean=0.0; std=1.0},
        [ordered]@{kind=3; value_index=0; mean=0.0; std=1.0},
        [ordered]@{kind=4; value_index=0; mean=0.0; std=1.0}
    )
}
$deployment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'deployment.json') -Encoding utf8NoBOM
Write-Host "Staged inference deployment: $output"
