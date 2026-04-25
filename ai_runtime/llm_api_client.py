import json
import time
from openai import AzureOpenAI, OpenAI
from ai_runtime.config_api import cfg
from ai_runtime.tts_client import VOICE_CATALOG

# Generate available voice list string for the system prompt
def _build_voice_list_str() -> str:
    parts = []
    for key, info in VOICE_CATALOG.items():
        parts.append(f"{key}({info['desc']})")
    return ", ".join(parts)

SYSTEM_PROMPT_JSON = (
    "You are a JSON generator for a VTuber system. "
    "You MUST output ONLY a single JSON object and nothing else. "
    "You can speak both English and Chinese. "
    "If the user asks in Chinese, reply in Chinese. If they ask in English, reply in English. "
    "Keep replies SHORT and conversational: ideally 15-30 Chinese characters or 10-25 English words. "
    "Only when the user explicitly asks for a story, explanation, or detail, you may go up to 80 characters/words. "
    "Never pad with filler. Be punchy.\n"
    "Do NOT use filler interjections at the start of a sentence (no å—¯ã€å—¢ã€å“¼ã€å“¦ã€å•Šã€å‘µå‘µ, no 'um'/'uh'/'well'/'oh'/'hmm'). "
    "Start each segment directly with content. Filler words confuse the lip-sync.\n"
    "The JSON schema is:\n"
    "{\n"
    '  "segments": [\n'
    '{"text": string, "emotion": "neutral|happy|sad|angry|surprised|excited|thinking|serious", "intensity": 0.0-1.0},\n'
    '    ... (1 to 5 segments)\n'
    '  ],\n'
    '  "motion_hint": "none|nod|shake|tilt_left|tilt_right",\n'
    '  "voice_change": string or null\n'
    "}\n"
    "\n"
    "IMPORTANT about segments:\n"
    "- Split your reply into 1-5 segments based on emotional shifts.\n"
    "- The concatenation of all segment texts IS your full reply.\n"
    "- Each segment should be a complete clause or sentence.\n"
    "- If the reply has a single consistent emotion, use just 1 segment.\n"
    "- Vary emotions naturally: e.g. start thinking, then become happy when you find the answer.\n"
    "- Use 'excited' for enthusiasm, delight, or energetic narration. Use 'surprised' ONLY for genuine shock or unexpected events.\n"
    "\n"
    "About voice_change:\n"
    "- When the user asks to change voice/éŸ³è‰²/å£°éŸ³, set voice_change to one of the available voice keys below.\n"
    "- When the user does NOT mention changing voice, set voice_change to null.\n"
    "- If the user asks to list available voices, put the list in reply and set voice_change to null.\n"
    "- If the user wants to go back to auto/default, set voice_change to \"auto\".\n"
    f"Available voices: {_build_voice_list_str()}\n"
    "\n"
    "Do NOT add any extra text outside JSON."
)

DEFAULT_RESPONSE = {
    "segments": [{"text": "I am not sure how to respond right now.", "emotion": "neutral", "intensity": 0.5}],
    "motion_hint": "none"
}


_VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry",
    "surprised", "excited", "thinking", "serious",
}


def _salvage_segments(raw: str) -> list[dict]:
    """ä»Žè¢«æˆªæ–­/ç ´æŸçš„ JSON ä¸²é‡ŒæŠ¢æ•‘å‡ºå·²é—­åˆçš„ segment é¡¹ã€‚

    æ€è·¯ï¼šç”¨ä¸€ä¸ªéžè´ªå©ªæ­£åˆ™æ‰« ``{"text": "...", "emotion": "...", "intensity": ...}``
    å½¢æ€çš„ objectï¼ˆå¿…é¡»ä¸‰ä¸ª key éƒ½é½æ‰ç®—ä¸€æ®µå®Œæ•´ï¼‰ï¼Œé€ä¸ª ``json.loads`` è§£æžï¼Œ
    æˆåŠŸä¸” emotion åˆæ³•çš„ç•™ä¸‹ã€‚è¿™æ ·æœ€åŽä¸€æ®µè¢« ``max_tokens`` æˆªæ–­æ—¶ï¼Œå‰é¢
    å®Œå¥½çš„å‡ æ®µä»å¯æ­£å¸¸æ’­æ”¾ã€‚
    """
    import re
    out: list[dict] = []
    # æ–‡æœ¬é‡Œå…è®¸æœ‰è½¬ä¹‰å¼•å·å’Œæ¢è¡Œï¼›ç”¨ DOTALL è®© . åŒ¹é…æ¢è¡Œã€‚
    pattern = re.compile(
        r'\{\s*"text"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*'
        r'"emotion"\s*:\s*"[^"]*"\s*,\s*'
        r'"intensity"\s*:\s*[0-9.]+\s*\}',
        re.DOTALL,
    )
    for m in pattern.finditer(raw):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        text = (obj.get("text") or "").strip()
        if not text:
            continue
        emo = obj.get("emotion") or "neutral"
        if emo not in _VALID_EMOTIONS:
            emo = "neutral"
        try:
            inten = float(obj.get("intensity", 0.5))
        except (TypeError, ValueError):
            inten = 0.5
        inten = max(0.0, min(1.0, inten))
        out.append({"text": text, "emotion": emo, "intensity": inten})
    return out

class LLMApiClient:
    def __init__(self, max_history: int = 20):
        if cfg.provider == "qwen":
            # Qwen via DashScope OpenAI 兼容模式
            self.client = OpenAI(
                base_url=cfg.endpoint,
                api_key=cfg.api_key,
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=cfg.endpoint,
                api_key=cfg.api_key,
                api_version=cfg.api_version,
            )
        self._history: list = []
        self._max_history = max_history

    def chat(self, user_text: str, history=None, max_retries: int = 3) -> dict:
        # ä½¿ç”¨å†…éƒ¨åŽ†å²è®°å½•ï¼ˆå¦‚æžœæ²¡æœ‰å¤–éƒ¨ä¼ å…¥ï¼‰
        if history is None:
            history = self._history
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_JSON},
            *history,
            {"role": "user", "content": user_text}
        ]
        
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=cfg.deployment,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=600,
                )
                # æ£€æŸ¥å†…å®¹è¿‡æ»¤ / æ‹’ç»
                choice = resp.choices[0] if resp.choices else None
                if choice is None:
                    print(f"[LLM] ç©ºå“åº”ï¼Œé‡è¯• {attempt+1}/{max_retries}")
                    continue
                finish = getattr(choice, "finish_reason", None)
                raw = (choice.message.content or "").strip()
                if not raw or finish == "content_filter":
                    print(f"[LLM] å“åº”è¢«è¿‡æ»¤æˆ–ä¸ºç©º (finish_reason={finish})ï¼Œé‡è¯• {attempt+1}/{max_retries}")

                    continue
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[LLM] è¿žæŽ¥å¤±è´¥ ({e.__class__.__name__}), {wait}s åŽé‡è¯•...")
                    time.sleep(wait)
                else:
                    print(f"[LLM] é‡è¯• {max_retries} æ¬¡ä»å¤±è´¥: {e}")
                    return DEFAULT_RESPONSE
        else:
            # æ‰€æœ‰é‡è¯•éƒ½å¤±è´¥
            print("[LLM] æ‰€æœ‰é‡è¯•å‡å¤±è´¥ï¼Œä½¿ç”¨é»˜è®¤å“åº”")
            return DEFAULT_RESPONSE
        
        raw = (resp.choices[0].message.content or "").strip()

        # Try to parse the response as JSON, with error handling
        try:
            # æœ‰æ—¶ LLM ä¼šç”¨ ```json ... ``` åŒ…è£¹
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            # å‘åŽå…¼å®¹ï¼šæ—§æ ¼å¼ {reply, emotion, intensity} â†’ æ–°æ ¼å¼ {segments}
            if "reply" in data and "segments" not in data:
                data["segments"] = [{
                    "text": data["reply"],
                    "emotion": data.get("emotion", "neutral"),
                    "intensity": float(data.get("intensity", 0.5)),
                }]
            # éªŒè¯ segments æ ¼å¼
            if "segments" not in data or not isinstance(data["segments"], list) or len(data["segments"]) == 0:
                raise ValueError("Missing or empty segments")
            # æž„å»º replyï¼ˆæ–¹ä¾¿å¤–éƒ¨ä½¿ç”¨ï¼‰
            data["reply"] = "".join(seg["text"] for seg in data["segments"])
            # ä¿å­˜å¯¹è¯åŽ†å²ï¼ˆåªä¿ç•™æ–‡æœ¬ï¼Œä¸ä¿å­˜æŽ§åˆ¶å­—æ®µï¼‰
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": raw})
            # one pair for each conversation, we keep the most recent 5 pairs
            if len(self._history) > self._max_history * 2:
                self._history = self._history[-self._max_history * 2:]
            return data
        except Exception as e:
            # JSON æˆªæ–­/ä¸åˆæ³•æ—¶ï¼Œå°è¯•æŠ¢æ•‘ï¼šç”¨æ­£åˆ™æå–å·²é—­åˆçš„ segment é¡¹ï¼Œ
            # é¿å…é•¿æ•…äº‹è¢« max_tokens æˆªåˆ°ä¸€åŠå°±ç›´æŽ¥èµ°"I am not sure..."å…œåº•ã€‚
            salvaged = _salvage_segments(raw)
            if salvaged:
                print(f"[LLM] JSON è§£æžå¤±è´¥ ({e})ï¼ŒæŠ¢æ•‘åˆ° {len(salvaged)} æ®µ")
                data = {
                    "segments": salvaged,
                    "motion_hint": "none",
                    "reply": "".join(s["text"] for s in salvaged),
                }
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": json.dumps(
                    {"segments": salvaged, "motion_hint": "none"}, ensure_ascii=False
                )})
                if len(self._history) > self._max_history * 2:
                    self._history = self._history[-self._max_history * 2:]
                return data
            print(f"[LLM] JSON è§£æžå¤±è´¥: {e}\n  raw={raw[:200]}")
            return DEFAULT_RESPONSE

    # Useless, can be deleted later
    def chat_once(self, text: str) -> str:
        resp = self.client.chat.completions.create(
            model=cfg.deployment,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
