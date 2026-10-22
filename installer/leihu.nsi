; 雷虎練唱台 安裝程式
Unicode true
ManifestDPIAware true
!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"

!define APPNAME "雷虎練唱台"
!define VERSION "1.5.3"
!define REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\LeihuLiantai"

Name "${APPNAME} ${VERSION}"
OutFile "out\雷虎練唱台1.5.3.exe"
InstallDir "$LOCALAPPDATA\LeihuLiantai"
InstallDirRegKey HKCU "${REGKEY}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "${APPNAME} ${VERSION}"
VIProductVersion "1.5.3.0"
VIAddVersionKey /LANG=1028 "ProductName" "${APPNAME}"
VIAddVersionKey /LANG=1028 "FileDescription" "${APPNAME} ${VERSION} 安裝程式"
VIAddVersionKey /LANG=1028 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1028 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1028 "LegalCopyright" "© 2026 雷虎肯茲 (Leihustore)"

!define MUI_ICON "payload\static\icon.ico"
!define MUI_UNICON "payload\static\icon.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "安裝 ${APPNAME} ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "雷虎練唱台會把歌曲分成主唱、合音、伴奏，畫出主唱旋律並辨識歌詞，讓你調 KEY、放慢速度練唱。$\r$\n$\r$\n安裝時會自動下載 Python、AI 套件與模型，約 3–5 GB，依網路速度需要 15–40 分鐘。有 NVIDIA 顯示卡時會自動安裝顯示卡加速版。$\r$\n$\r$\n安裝過程會開啟一個黑色視窗顯示進度，請不要關閉它。"
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "開啟雷虎練唱台"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\使用說明.html"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "閱讀使用說明"

!insertmacro MUI_PAGE_WELCOME
!define MUI_LICENSEPAGE_TEXT_TOP "請閱讀以下聲明。安裝即表示您同意這些內容。"
!define MUI_LICENSEPAGE_TEXT_BOTTOM "同意以上內容請按「我同意」繼續安裝。"
!insertmacro MUI_PAGE_LICENSE "license.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "TradChinese"

Var PS

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "雷虎練唱台需要 64 位元的 Windows 10 或 11。"
    Abort
  ${EndIf}
FunctionEnd

Function FindPowerShell
  ; 32 位元的安裝程式要透過 Sysnative 才能開 64 位元的 PowerShell
  StrCpy $PS "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
  ${IfNot} ${FileExists} "$PS"
    StrCpy $PS "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"
  ${EndIf}
FunctionEnd

!macro StopApp DIR
  StrCpy $PS "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
  ${IfNot} ${FileExists} "$PS"
    StrCpy $PS "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"
  ${EndIf}
  nsExec::Exec `"$PS" -NoProfile -NonInteractive -Command "Get-CimInstance Win32_Process | Where-Object { $$_.ExecutablePath -like '${DIR}\python\*' } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force }"`
  Pop $0
  Sleep 1500
!macroend

Function LaunchApp
  Exec '"$INSTDIR\python\pythonw.exe" "$INSTDIR\desktop.py"'
FunctionEnd

Section "Install"
  AddSize 7000000
  ; 偵測之前的安裝：完整安裝過 → 問要更新還是全新重裝；裝到一半中斷 → 問要接著修復還是清除重裝
  ${If} ${FileExists} "$INSTDIR\python\python.exe"
    ${If} ${FileExists} "$INSTDIR\installed.json"
      MessageBox MB_YESNOCANCEL|MB_ICONQUESTION "這台電腦已經安裝過雷虎練唱台。$\r$\n$\r$\n「是」：更新（沿用已下載的套件與模型，最快）$\r$\n「否」：全新重裝（清除舊的程式與套件，重新下載）$\r$\n「取消」：不安裝$\r$\n$\r$\n兩種方式都會保留歌曲庫、錄音與設定。" IDYES keep IDNO wipe
    ${Else}
      MessageBox MB_YESNOCANCEL|MB_ICONEXCLAMATION "偵測到上次安裝沒有完成。$\r$\n$\r$\n「是」：接著安裝，並自動檢查、修復不完整的檔案（建議）$\r$\n「否」：清除上次留下的檔案後全新安裝$\r$\n「取消」：不安裝$\r$\n$\r$\n兩種方式都會保留歌曲庫、錄音與設定。" IDYES keep IDNO wipe
    ${EndIf}
    Abort "已取消安裝"
  wipe:
    !insertmacro StopApp "$INSTDIR"
    DetailPrint "清除舊的程式與套件…"
    RMDir /r "$INSTDIR\python"
    RMDir /r "$INSTDIR\downloads"
    RMDir /r "$INSTDIR\static"
    RMDir /r "$INSTDIR\setup"
    RMDir /r "$INSTDIR\bin"
    RMDir /r "$INSTDIR\__pycache__"
    RMDir /r "$INSTDIR\.webview"
    Delete "$INSTDIR\*.py"
    Delete "$INSTDIR\requirements*.txt"
    Delete "$INSTDIR\installed.json"
  keep:
  ${EndIf}
  !insertmacro StopApp "$INSTDIR"
  SetOutPath "$INSTDIR"
  File /r "payload\*"

  DetailPrint "下載並安裝 Python、AI 套件與模型（會開啟一個進度視窗）…"
  Call FindPowerShell
  ExecWait '"$PS" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\setup\install.ps1" -AppDir "$INSTDIR"' $0
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION "安裝沒有完成。$\r$\n$\r$\n請確認網路連線後重新執行安裝程式，選「是」接著安裝，已下載的部分不會重來。$\r$\n詳細紀錄：$INSTDIR\logs\install.log"
    Abort "安裝沒有完成"
  ${EndIf}

  ; 捷徑
  CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\desktop.py"' "$INSTDIR\static\icon.ico" 0
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\desktop.py"' "$INSTDIR\static\icon.ico" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME} 修復與更新.lnk" "$PS" '-NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\setup\install.ps1" -AppDir "$INSTDIR" -Repair' "$INSTDIR\static\icon.ico" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME} 修復（強制使用顯示卡加速）.lnk" "$PS" '-NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\setup\install.ps1" -AppDir "$INSTDIR" -Repair -Gpu cu128' "$INSTDIR\static\icon.ico" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME} 使用說明.lnk" "$INSTDIR\使用說明.html"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\解除安裝 ${APPNAME}.lnk" "$INSTDIR\uninstall.exe"

  ; 「應用程式」清單中的解除安裝項目
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "${REGKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${REGKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${REGKEY}" "Publisher" "雷虎肯茲 (Leihustore)"
  WriteRegStr HKCU "${REGKEY}" "DisplayIcon" "$INSTDIR\static\icon.ico"
  WriteRegStr HKCU "${REGKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${REGKEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKCU "${REGKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${REGKEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${REGKEY}" "EstimatedSize" 7000000
SectionEnd

Section "Uninstall"
  !insertmacro StopApp "$INSTDIR"
  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$SMPROGRAMS\${APPNAME}"
  DeleteRegKey HKCU "${REGKEY}"
  MessageBox MB_YESNO|MB_ICONQUESTION "要一併刪除歌曲庫、錄音與設定（含 API 金鑰）嗎？$\r$\n$\r$\n選「否」會保留在：$INSTDIR\library" IDYES all
    ; 保留歌曲庫：只移除程式
    RMDir /r "$INSTDIR\python"
    RMDir /r "$INSTDIR\downloads"
    RMDir /r "$INSTDIR\models"
    RMDir /r "$INSTDIR\static"
    RMDir /r "$INSTDIR\setup"
    RMDir /r "$INSTDIR\bin"
    RMDir /r "$INSTDIR\logs"
    RMDir /r "$INSTDIR\__pycache__"
    RMDir /r "$INSTDIR\.webview"
    Delete "$INSTDIR\*.py"
    Delete "$INSTDIR\*.txt"
    Delete "$INSTDIR\*.html"
    Delete "$INSTDIR\VERSION"
    Delete "$INSTDIR\installed.json"
    Delete "$INSTDIR\uninstall.exe"
    Goto done
  all:
    RMDir /r "$INSTDIR"
  done:
SectionEnd
