# 바탕화면 바로가기 생성.
#
# .lnk 로 만드는 이유: exe 로 감싸면 사용자가 받을 때 백신 경고가 뜨고
# (서명 없는 PyInstaller 산출물은 흔히 오탐된다), 파이썬 인터프리터까지
# 300MB 를 함께 배포해야 한다. 바로가기는 서명이 필요 없고 즉시 만들어진다.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$icon = Join-Path $root "assets\regime-quant.ico"
$target = Join-Path $root "windows\start.bat"

if (-not (Test-Path $target)) { throw "start.bat 을 찾을 수 없습니다: $target" }

$desktop = [Environment]::GetFolderPath("Desktop")
$link = Join-Path $desktop "Regime Quant.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath       = $target
$sc.WorkingDirectory = $root
$sc.Description      = "레짐 기반 암호화폐 백테스트·신호 조회 (주문 기능 없음)"
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.WindowStyle      = 1
$sc.Save()

Write-Host ""
Write-Host "  바탕화면에 'Regime Quant' 아이콘을 만들었습니다." -ForegroundColor Green
Write-Host "  $link"
Write-Host ""
