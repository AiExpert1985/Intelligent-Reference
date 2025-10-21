# utils/highlight_token.py
import base64, hmac, json, time
from hashlib import sha256
from typing import Dict, Any
from config import settings

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def _b64d(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)

def sign(payload: Dict[str, Any], exp_seconds: int = 120) -> str:
    header = {"alg":"HS256","typ":"HLT"}
    body = dict(payload); body["exp"] = int(time.time()) + exp_seconds
    h = _b64(json.dumps(header, separators=(",",":")).encode())
    p = _b64(json.dumps(body,   separators=(",",":")).encode())
    mac = hmac.new(settings.HIGHLIGHT_TOKEN_SECRET.encode(), f"{h}.{p}".encode(), sha256).digest()
    s = _b64(mac)
    return f"{h}.{p}.{s}"

def verify(token: str) -> Dict[str, Any]:
    try:
        h,p,s = token.split(".")
        expected = hmac.new(settings.HIGHLIGHT_TOKEN_SECRET.encode(), f"{h}.{p}".encode(), sha256).digest()
        got = _b64d(s)
        if not hmac.compare_digest(expected, got):
            raise ValueError
        body = json.loads(_b64d(p))
        if int(time.time()) > int(body.get("exp",0)):
            raise ValueError
        return body
    except Exception:
        raise ValueError("invalid_or_expired_token")
