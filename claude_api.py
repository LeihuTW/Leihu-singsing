"""練唱台：呼叫 Claude（需要連網），並記錄每次用掉的 token 與費用。"""
from __future__ import annotations

import json
import os
import threading
import time

import pipeline as P

USAGE_PATH = P.ROOT / "usage.json"
_lock = threading.Lock()
SESSION_START = time.time()

# 每百萬 token 的美元價格（輸入、輸出），來源：Anthropic 官方價目表。
# 依模型 ID 前綴比對；不在表上的模型只顯示 token 數。
PRICES = [
    ("claude-haiku-5-5", 0.10, 0.50),     # 提示 10 萬 token 以下的價格
    ("claude-sonnet-5-5", 2.0, 10.0),
    ("claude-opus-5-5", 4.0, 20.0),
    ("claude-fable-5-1", 10.0, 50.0),
    ("claude-mythos-5-1", 10.0, 50.0),
    ("claude-sonnet-5", 2.0, 10.0),
    ("claude-opus-5", 5.0, 25.0),
    ("claude-fable-5", 10.0, 50.0),
    ("claude-opus-4-8", 5.0, 25.0),
    ("claude-opus-4-7", 5.0, 25.0),
    ("claude-opus-4-6", 5.0, 25.0),
    ("claude-opus-4-5", 5.0, 25.0),
    ("claude-sonnet-4", 3.0, 15.0),
    ("claude-haiku-4-5", 1.0, 5.0),
]
PURPOSES = {"align": "歌詞對齊", "coach": "練習建議"}


def price_for(model: str, input_tokens: int):
    for prefix, pin, pout in PRICES:
        if model.startswith(prefix):
            if prefix == "claude-haiku-5-5" and input_tokens > 100_000:
                return 0.50, 2.50
            return pin, pout
    return None


def _load() -> dict:
    try:
        return json.loads(USAGE_PATH.read_text("utf-8"))
    except Exception:
        return {"calls": []}


def record(model: str, usage, purpose: str, song: str) -> dict:
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    pr = price_for(model, tin)
    cost = round((tin * pr[0] + tout * pr[1]) / 1_000_000, 6) if pr else None
    entry = {"time": time.time(), "model": model, "purpose": purpose, "song": song,
             "input_tokens": tin, "output_tokens": tout, "cost_usd": cost}
    with _lock:
        data = _load()
        data["calls"].append(entry)
        data["calls"] = data["calls"][-2000:]
        tmp = USAGE_PATH.with_name(USAGE_PATH.name + ".tmp")   # 已經在 _lock 裡，同一時間只有一個人寫
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(USAGE_PATH)
    return entry


def _sum(calls) -> dict:
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "cost_usd": round(sum(c["cost_usd"] or 0 for c in calls), 6),
        "unpriced": sum(1 for c in calls if c["cost_usd"] is None),
    }


def summary() -> dict:
    cfg = P.load_config()
    calls = _load()["calls"]
    out = {
        "last": calls[-1] if calls else None,
        "session": _sum([c for c in calls if c["time"] >= SESSION_START]),
        "total": _sum(calls),
        "recent": calls[-8:][::-1],
        "balance": None,
    }
    bal, since = cfg.get("balance_usd"), cfg.get("balance_set_at")
    if bal not in (None, "") and since:
        spent = _sum([c for c in calls if c["time"] >= since])["cost_usd"]
        out["balance"] = {"entered_usd": float(bal), "entered_at": since, "spent_since_usd": spent,
                          "estimated_usd": round(float(bal) - spent, 4)}
    return out


# ---------------------------------------------------------------- 呼叫

def _client():
    cfg = P.load_config()
    key = cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise P.UserError("還沒設定 Anthropic API 金鑰。到「設定」貼上金鑰後再試。")
    try:
        import anthropic
    except ImportError:
        raise P.UserError("沒有安裝 anthropic 套件")
    return anthropic, anthropic.Anthropic(api_key=key), cfg.get("claude_model") or "claude-sonnet-5-5"


def _call(kwargs: dict, purpose: str, song: str, no_think: bool = False):
    anthropic, client, model = _client()
    try:
        if no_think:
            # 1.6.5：新模型預設會先思考，對齊歌詞這種只要回傳數字的工作，思考會把 token 用光、回覆被截斷。
            # between_tools＝不先思考、直接呼叫工具；舊模型不認得這個設定時（400 錯誤，不計費）改用一般方式。
            try:
                msg = client.messages.create(model=model, extra_body={"thinking": {"type": "between_tools"}}, **kwargs)
            except anthropic.BadRequestError as e:
                if "thinking" not in str(e.message).lower():
                    raise
                msg = client.messages.create(model=model, **kwargs)
        else:
            msg = client.messages.create(model=model, **kwargs)
    except anthropic.APIConnectionError:
        raise P.UserError("連不上 Claude。確認網路後再試。")
    except anthropic.AuthenticationError:
        raise P.UserError("API 金鑰無效。到「設定」重新貼上。")
    except anthropic.NotFoundError:
        raise P.UserError(f"找不到模型「{model}」。到「設定」確認 Claude 模型名稱。")
    except anthropic.APIStatusError as e:
        if "credit balance" in str(e.message).lower():
            raise P.UserError("API 額度不足。請到 Anthropic Console 的 Billing 頁面加值。")
        raise P.UserError(f"Claude 回應錯誤（{e.status_code}）：{e.message}")
    entry = record(model, msg.usage, purpose, song)
    return msg, entry


def ask_text(prompt: str, max_tokens: int, purpose: str, song: str = "") -> tuple[str, dict]:
    """回傳文字。回覆被長度上限截斷時，自動用更大的上限重問一次（模型可能先花掉一部分在思考）。"""
    msg, entry = _call({"max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]},
                       purpose, song)
    if getattr(msg, "stop_reason", "") == "max_tokens":
        log_raw(purpose, msg)
        msg, e2 = _call({"max_tokens": max_tokens * 4, "messages": [{"role": "user", "content": prompt}]},
                        purpose, song)
        entry = {**e2,
                 "input_tokens": entry["input_tokens"] + e2["input_tokens"],
                 "output_tokens": entry["output_tokens"] + e2["output_tokens"],
                 "cost_usd": None if entry["cost_usd"] is None or e2["cost_usd"] is None
                 else round(entry["cost_usd"] + e2["cost_usd"], 6)}
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if getattr(msg, "stop_reason", "") == "max_tokens":
        text += "……\n（回覆太長被截斷，可以再按一次重新分析）"
    return text, entry


def ask_tool(prompt: str, tool: dict, max_tokens: int, purpose: str, song: str = "") -> tuple[dict, dict]:
    """請 Claude 用指定工具回傳結構化資料。

    有些模型不支援強制指定工具（tool_choice），所以只提供工具並在提示中要求使用；
    萬一 Claude 改用文字回答，再從文字裡找 JSON。
    """
    msg, entry = _call({
        "max_tokens": max_tokens,
        "tools": [tool],
        "messages": [{"role": "user", "content": prompt + f"\n\n請直接呼叫 {tool['name']} 工具回傳結果，不要先用文字說明或推理。"}],
    }, purpose, song, no_think=True)
    for b in msg.content:
        if getattr(b, "type", "") == "tool_use" and b.name == tool["name"]:
            return dict(b.input), entry
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
    data = _json_in_text(text)
    if data is not None:
        return data, entry
    log_raw(purpose, msg)
    if getattr(msg, "stop_reason", "") == "max_tokens":
        raise P.UserError("Claude 的回覆被截斷（已用掉的費用見下方）。可以再試一次，或分段貼上再對齊。")
    raise P.UserError("Claude 沒有回傳時間資料，再試一次。")


def _json_in_text(text: str):
    import re
    for m in re.finditer(r"\{.*\}", text, re.S):
        try:
            return json.loads(m.group(0))
        except ValueError:
            continue
    m = re.search(r"\[[\s\d.,]+\]", text)
    if m:
        try:
            return {"times": json.loads(m.group(0))}
        except ValueError:
            pass
    return None


def log_raw(purpose: str, msg) -> None:
    try:
        (P.ROOT / "logs").mkdir(exist_ok=True)
        with open(P.ROOT / "logs" / "claude.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {purpose} stop={getattr(msg, 'stop_reason', '')}\n")
            f.write(repr(msg.content)[:4000] + "\n")
    except Exception:
        pass
