# 雷虎練唱台

在自己電腦上跑的 Windows 練歌工具。匯入一首歌（音訊檔或 YouTube 網址），它會：

- 用 AI 模型把歌拆成**主唱、合音、伴奏**三軌
- 分析主唱的**旋律**，可切換「原聲曲線」或「音符區塊」顯示
- **辨識歌詞**並對上時間，中文轉成台灣繁體；也會到 LRCLIB、網易雲、酷狗搜尋歌詞，與辨識結果比對後讓你選一版；也可以匯入 LRC／SRT／TXT 歌詞檔
- **升降 KEY、調整速度**練唱（用所有 CPU 核心轉換）；唱的時候會依你的音域**建議適合的 KEY**
- 可以關掉旋律只看歌詞（五行）；也有給 OBS 用的**去背直播歌詞**畫面，跟著播放同步上色
- 開麥克風時即時顯示你的音高，有音量表、增益與自動調整、防爆音；錄音後在本機**自動評分**，標出最需要練的句子
- 把你的錄音和合音、伴奏混音後**匯出 MP3／WAV、直式影片（旋律線條＋歌詞）或成績卡圖片**，一鍵開啟 FB、IG、X、Threads 分享
- 歌曲庫可自訂分類（例如中文歌、日文歌）並用關鍵字搜尋

歌曲處理都在本機完成（有 NVIDIA 顯示卡時用 GPU 加速）。只有兩個功能會連網使用 Claude，而且需要自己的 Anthropic API 金鑰：「Claude 精準對齊歌詞」和「練習建議」。

## 安裝

到 [Releases](../../releases) 下載最新版的 `LeihuLiantai-<版本>-Setup.exe`（雷虎練唱台安裝檔），雙擊安裝。安裝程式會自動：

1. 下載內嵌版 Python 3.11（不影響電腦上原本的 Python）
2. 偵測顯示卡，安裝對應的 PyTorch（CUDA 12.8／12.6 或 CPU 版）
3. 下載分離與辨識模型（約 2 GB）
4. 建立桌面與開始選單捷徑

安裝視窗會顯示總進度和下載速度。第一次安裝需要連網，約 10–30 分鐘，之後可以離線使用。安裝中斷時重新執行安裝檔即可，會接著下載並自動修復不完整的檔案。詳細操作見 [使用說明](static/help.html)。

系統需求：Windows 10/11 64 位元、約 8 GB 磁碟空間；建議 NVIDIA 顯示卡（驅動支援 CUDA 12.6 以上）。

## 從原始碼建置安裝檔

需要 [NSIS](https://nsis.sourceforge.io/)：

```sh
sh installer/build.sh
```

產生的安裝檔在 `installer/out/`。

## 檔案結構

| 檔案 | 用途 |
| --- | --- |
| `app.py` | 本機伺服器（Flask） |
| `desktop.py` | 桌面視窗（pywebview），伺服器在另一個行程執行 |
| `pipeline.py` | 分軌、音高、歌詞辨識、線上歌詞搜尋、歌曲庫 |
| `claude_api.py` | 呼叫 Claude 並記錄 token 用量與費用 |
| `share.py` | 錄音混音、匯出音訊、直式影片與成績卡 |
| `installer.py` | 安裝時偵測顯示卡、安裝套件、下載模型 |
| `setup/install.ps1` | 安裝檔呼叫的 PowerShell 安裝腳本 |
| `installer/leihu.nsi` | NSIS 安裝檔腳本 |
| `static/` | 介面（HTML／JS／CSS）與使用說明 |

API 金鑰、歌曲庫、用量紀錄都存在安裝資料夾裡的 `config.json`、`library/`、`usage.json`，不會被提交到這個 repo（見 `.gitignore`）。

## 使用的開源專案

[audio-separator](https://github.com/nomadkaraoke/python-audio-separator)（UVR MDX-Net 模型）、[torchcrepe](https://github.com/maxrporter/torchcrepe)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[OpenCC](https://github.com/BYVoid/OpenCC)、[pedalboard](https://github.com/spotify/pedalboard)、[yt-dlp](https://github.com/yt-dlp/yt-dlp)、[pywebview](https://github.com/r0x0r/pywebview)、[LRCLIB](https://lrclib.net/)。

## 授權

雷虎練唱台 © 2026 雷虎肯茲（Leihustore）。授權條款見 [LICENSE.md](LICENSE.md)（PolyForm Noncommercial 1.0.0，加上作者的附加許可）：

- **可以**：個人練唱、學習，以及實況直播使用（包含有訂閱、贊助收入的實況）
- **可以**：修改並免費分享原版或修改版，但須註明來源（雷虎練唱台／雷虎肯茲 Leihustore，附上本專案網址）並保留 LICENSE.md
- **不可以**：把本工具或修改後的版本打包販售、收費散布，或做成付費商品、付費服務

本工具使用 AI 開發，並調度多項 AI 工具協作，可能發生不明狀況。免費提供，安裝或使用即視同同意：使用過程中發生的任何狀況與損害由使用者自行承擔，開發者不負損害賠償責任。

使用的第三方套件和模型各有自己的授權。匯入的歌曲與歌詞版權屬於原權利人，請只匯入你有權使用的音樂。
