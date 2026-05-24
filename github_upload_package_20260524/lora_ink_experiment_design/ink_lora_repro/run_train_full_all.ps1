$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\38230\Desktop\1\ink_lora_repro"
$HfEndpoint = "https://hf-mirror.com"

Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

$Configs = @(
  ".\configs\experiments_full\paper_r08_s1200_6gb_full.json",
  ".\configs\experiments_full\paper_r16_s600_6gb_full.json",
  ".\configs\experiments_full\paper_r16_s1200_6gb_full.json",
  ".\configs\experiments_full\paper_r16_s1800_6gb_full.json",
  ".\configs\experiments_full\paper_r32_s1200_6gb_full.json"
)

foreach ($Config in $Configs) {
  Write-Host "Training with config: $Config"
  python .\scripts\train_lora.py --config $Config --hf_endpoint $HfEndpoint
}

Write-Host "Full-dataset five-group training finished."
