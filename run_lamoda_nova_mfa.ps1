# MFA -> temp AWS creds -> add_descriptions_nova.py (full Lamoda women)
# Run:   .\run_lamoda_nova_mfa.ps1
# Or:   .\run_lamoda_nova_mfa.ps1 -MfaCode 123456
# Keep this file ASCII-only so Windows PowerShell 5.1 parses it reliably.

param(
    [Parameter(Mandatory = $false)]
    [string] $MfaCode
)

$ErrorActionPreference = "Stop"
$MfaSerial = "arn:aws:iam::861032082686:mfa/gleb-mfa"
$AwsProfile = "modera"
$Region = "us-east-1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $MfaCode) {
    $MfaCode = Read-Host "MFA code (6 digits)"
}
$MfaCode = $MfaCode.Trim()
if ($MfaCode -notmatch '^\d{6}$') {
    Write-Error "MFA code must be exactly 6 digits."
    exit 1
}

Write-Host "Getting temporary AWS session..."
$raw = aws sts get-session-token `
    --serial-number $MfaSerial `
    --token-code $MfaCode `
    --duration-seconds 43200 `
    --profile $AwsProfile `
    --region $Region `
    --output json 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "get-session-token failed: $raw"
    exit $LASTEXITCODE
}

$r = $raw | ConvertFrom-Json
$env:AWS_ACCESS_KEY_ID = $r.Credentials.AccessKeyId
$env:AWS_SECRET_ACCESS_KEY = $r.Credentials.SecretAccessKey
$env:AWS_SESSION_TOKEN = $r.Credentials.SessionToken
$env:AWS_DEFAULT_REGION = $Region

Write-Host "Session expires: $($r.Credentials.Expiration)"
Set-Location $ScriptDir

python add_descriptions_nova.py `
    --no-profile `
    --input lamoda_women.cleaned.json `
    --output lamoda_women_with_descriptions_nova.json `
    --limit 15700 `
    --images-dir ./images_lamoda_nova

exit $LASTEXITCODE
