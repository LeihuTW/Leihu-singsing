#!/bin/sh
# 建立 Windows 安裝檔：需要 NSIS（makensis）。在 repo 根目錄執行 sh installer/build.sh
set -e
cd "$(dirname "$0")"
rm -rf payload && mkdir -p payload/static payload/setup out
cp ../app.py ../desktop.py ../installer.py ../pipeline.py ../claude_api.py ../share.py ../VERSION ../requirements-win.txt payload/
cp ../static/index.html ../static/app.js ../static/style.css ../static/help.html ../static/overlay.html ../static/icon.ico ../static/kenzi.png ../static/linepay.png payload/static/
cp ../static/help.html payload/使用說明.html
cp ../setup/install.ps1 payload/setup/
cp license.txt payload/版權與使用聲明.txt
LANG=C.UTF-8 LC_ALL=C.UTF-8 makensis -V2 leihu.nsi
ls -la out
