"""全域快捷鍵的定義（2.0）：桌面程式（desktop.py）註冊按鍵、伺服器（app.py）給設定畫面用，兩邊共用這份。

組合鍵用文字表示，例如 "Ctrl+Alt+Space"、"Ctrl+Shift+F5"、"F13"。
存在 config.json 的 "hotkeys"：{動作: 組合鍵}，空字串＝不使用；沒寫的動作用預設值。
"""
from __future__ import annotations

ACTIONS = [
    ("play", "播放／暫停", "Ctrl+Alt+Space"),
    ("now", "這句對準", "Ctrl+Alt+A"),
    ("next-line", "下一句對準", "Ctrl+Alt+S"),
    ("prev-line", "上一句對準", ""),
    ("minus", "目前這句提早 0.1 秒", "Ctrl+Alt+Comma"),
    ("plus", "目前這句延後 0.1 秒", "Ctrl+Alt+Period"),
    ("mode", "原曲／卡拉OK 切換", "Ctrl+Alt+K"),
    ("next-song", "待唱清單的下一首", "Ctrl+Alt+N"),
    ("key-up", "KEY 升一個半音", "Ctrl+Alt+PageUp"),
    ("key-down", "KEY 降一個半音", "Ctrl+Alt+PageDown"),
    ("flag", "標記／取消「歌詞待修」", "Ctrl+Alt+F"),
    ("pause-all", "處理佇列全部暫停／繼續", "Ctrl+Alt+P"),
]
ACTION_IDS = [a for a, _, _ in ACTIONS]

MODS = {"Ctrl": 0x0002, "Alt": 0x0001, "Shift": 0x0004, "Win": 0x0008}
KEYS: dict[str, int] = {
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(n): 0x30 + n for n in range(10)},
    **{f"F{n}": 0x70 + n - 1 for n in range(1, 25)},
    **{f"Num{n}": 0x60 + n for n in range(10)},
    "Space": 0x20, "Enter": 0x0D, "PageUp": 0x21, "PageDown": 0x22, "End": 0x23, "Home": 0x24,
    "Left": 0x25, "Up": 0x26, "Right": 0x27, "Down": 0x28, "Insert": 0x2D, "Delete": 0x2E,
    "NumMultiply": 0x6A, "NumAdd": 0x6B, "NumSubtract": 0x6D, "NumDecimal": 0x6E, "NumDivide": 0x6F,
    "Semicolon": 0xBA, "Equal": 0xBB, "Comma": 0xBC, "Minus": 0xBD, "Period": 0xBE, "Slash": 0xBF,
    "Backquote": 0xC0, "BracketLeft": 0xDB, "Backslash": 0xDC, "BracketRight": 0xDD, "Quote": 0xDE,
    "Pause": 0x13, "ScrollLock": 0x91,
}
# 不按修飾鍵也能單獨用的鍵：F13～F24（一般鍵盤沒有，Stream Deck／巨集鍵盤常用），避免搶走一般打字用的鍵
SOLO_OK = {f"F{n}" for n in range(13, 25)}


def parse(combo: str) -> tuple[int, int] | None:
    """"Ctrl+Alt+Space" → (修飾鍵旗標, 虛擬鍵碼)；格式不對回傳 None。"""
    parts = [p.strip() for p in str(combo or "").split("+") if p.strip()]
    if not parts:
        return None
    *mods, key = parts
    if key not in KEYS or len(set(mods)) != len(mods) or any(m not in MODS for m in mods):
        return None
    if not mods and key not in SOLO_OK:
        return None
    flag = 0
    for m in mods:
        flag |= MODS[m]
    return flag, KEYS[key]


def current(cfg: dict) -> dict[str, str]:
    """目前的設定：沒寫的動作用預設值，空字串＝不使用。"""
    saved = cfg.get("hotkeys") if isinstance(cfg.get("hotkeys"), dict) else {}
    return {a: (saved[a] if isinstance(saved.get(a), str) else d) for a, _, d in ACTIONS}
