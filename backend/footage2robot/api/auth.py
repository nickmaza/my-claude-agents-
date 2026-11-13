"""Firebase Authentication: verify ID tokens sent by the web app.

Set ``FOOTAGE2ROBOT_FIREBASE_PROJECT`` to the Firebase project ID to require
sign-in. Without it the server runs in single-user local mode (everything
belongs to the user ``local``), which is only appropriate on a trusted machine.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException

CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
LOCAL_USER = "local"


@dataclass
class User:
    uid: str
    email: str | None = None


def project_id() -> str | None:
    return os.environ.get("FOOTAGE2ROBOT_FIREBASE_PROJECT") or None


def web_config() -> dict | None:
    """Public Firebase web-app config handed to the browser (not a secret)."""
    raw = os.environ.get("FOOTAGE2ROBOT_FIREBASE_WEB_CONFIG")
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return cfg if isinstance(cfg, dict) and cfg.get("apiKey") else None


_certs: dict = {"value": None, "expires": 0.0}
_lock = threading.Lock()


def _google_certs() -> dict:
    with _lock:
        if _certs["value"] is None or time.time() > _certs["expires"]:
            import requests
            res = requests.get(CERTS_URL, timeout=10)
            res.raise_for_status()
            _certs.update(value=res.json(), expires=time.time() + 3600)
        return _certs["value"]


def verify_token(token: str) -> User:
    """Check signature, expiry, audience and issuer of a Firebase ID token."""
    from google.auth import jwt

    project = project_id()
    claims = jwt.decode(token, certs=_google_certs(), audience=project)
    if claims.get("iss") != f"https://securetoken.google.com/{project}" or not claims.get("sub"):
        raise ValueError("Wrong issuer or missing subject")
    return User(uid=str(claims["sub"]), email=claims.get("email"))


def current_user(authorization: str | None = Header(default=None)) -> User:
    if project_id() is None:
        return User(LOCAL_USER)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required", headers={"WWW-Authenticate": "Bearer"})
    try:
        return verify_token(authorization[7:].strip())
    except Exception as e:
        raise HTTPException(401, "Invalid or expired sign-in token", headers={"WWW-Authenticate": "Bearer"}) from e


def optional_user(authorization: str | None = Header(default=None)) -> User | None:
    """For file URLs, which may be opened with ?key= instead of a header."""
    if project_id() is None:
        return User(LOCAL_USER)
    if not authorization:
        return None
    return current_user(authorization)
