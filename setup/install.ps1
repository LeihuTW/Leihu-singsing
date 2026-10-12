# 雷虎練唱台 安裝：準備獨立的 Python，再交給 installer.py 安裝套件與模型。
param([Parameter(Mandatory=$true)][string]$AppDir, [switch]$Repair)
$ErrorActionPreference = "Continue"   # 外部程式寫到 stderr 不算錯誤；自己檢查結束代碼
$ProgressPreference = "SilentlyContinue"   # 關掉進度列，Invoke-WebRequest 會快很多
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "雷虎練唱台 安裝中…（請勿關閉）"
New-Item -ItemType Directory -Force -Path "$AppDir\logs" | Out-Null
$log = "$AppDir\logs\install.log"
function Say($m) { Write-Host $m; Add-Content -Path $log -Value $m -Encoding UTF8 }

function Get-File($url, $out) {
  for ($i = 1; $i -le 3; $i++) {
    try { Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -ErrorAction Stop; return }
    catch { Say "  下載失敗（第 $i 次）：$($_.Exception.Message)"; Start-Sleep -Seconds 3 }
  }
  throw "無法下載 $url"
}

try {
  Say ""
  Say "雷虎練唱台 安裝程式"
  Say "========================================"
  $py = Join-Path $AppDir "python\python.exe"
  if (-not (Test-Path $py)) {
    Say "下載 Python 3.11（約 11 MB）…"
    $tmp = Join-Path $env:TEMP "leihu-python.zip"
    Get-File "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" $tmp
    Expand-Archive -Path $tmp -DestinationPath (Join-Path $AppDir "python") -Force -ErrorAction Stop
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $py)) { throw "Python 解壓縮失敗" }
    # 讓這份 Python 能安裝套件，並找得到程式本體（上一層資料夾）
    Set-Content -Path (Join-Path $AppDir "python\python311._pth") -Encoding ASCII -Value @(
      "python311.zip", ".", "Lib\site-packages", "..", "import site")
  }
  $pipOk = $false
  & $py -m pip --version 2>&1 | Out-Null
  $pipOk = ($LASTEXITCODE -eq 0)
  if (-not $pipOk) {
    Say "安裝 pip…"
    $gp = Join-Path $AppDir "python\get-pip.py"
    Get-File "https://bootstrap.pypa.io/get-pip.py" $gp
    & $py $gp --no-warn-script-location 2>&1 | ForEach-Object { "$_" } | Add-Content -Path $log -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "pip 安裝失敗" }
  }
  chcp 65001 > $null
  $env:PYTHONIOENCODING = "utf-8"
  $args2 = @((Join-Path $AppDir "installer.py"))
  if ($Repair) { $args2 += @("--repair", "--launch", "--pause") }
  & $py @args2
  exit $LASTEXITCODE
}
catch {
  Say ""
  Say "安裝沒有完成：$($_.Exception.Message)"
  Say "請確認網路連線後重新執行安裝程式。紀錄在 $log"
  if ($Repair) { Read-Host "按 Enter 關閉" }
  exit 1
}
