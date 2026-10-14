"""
services/auth.py
================
Authentication service for Google OAuth 2.0 (Google Identity Services)
and lightweight HMAC-signed session tokens.
"""

import os
import hmac
import hashlib
import json
import base64
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("telecom_rag.auth")

SECRET_KEY = os.environ.get("SESSION_SECRET", "telecom_3gpp_rag_auth_secret_token_2026")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")


def verify_google_id_token(token_str: str) -> Optional[Dict[str, Any]]:
    """
    Verifies a Google ID Token received from Google Identity Services (GSI)
    using the official google-auth library.
    """
    if not token_str:
        return None

    # Development / Demo bypass for testing without active Google credentials
    if token_str.startswith("demo_token_") or token_str == "test_google_token":
        return {
            "google_id": "demo_google_id_12345",
            "email": "telecom_tester@domain.com",
            "name": "Telecom Standards Engineer",
            "picture": "https://lh3.googleusercontent.com/a/default-user"
        }

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests

        # If GOOGLE_CLIENT_ID is configured, verify_oauth2_token validates the audience.
        # Otherwise, it verifies cryptographic integrity, certificate signatures, and expiration.
        audience = GOOGLE_CLIENT_ID if GOOGLE_CLIENT_ID else None
        idinfo = id_token.verify_oauth2_token(token_str, requests.Request(), audience)

        # Token is valid; extract profile claims
        return {
            "google_id": str(idinfo.get("sub")),
            "email": str(idinfo.get("email")),
            "name": str(idinfo.get("name", "Telecom User")),
            "picture": str(idinfo.get("picture", "")),
        }
    except Exception as e:
        logger.error(f"[Auth] Google token verification error: {e}")
        return None


def create_session_token(user_id: int, email: str, name: str, picture: str = "", expires_in: int = 30 * 86400) -> str:
    """
    Creates a cryptographically signed HMAC-SHA256 session token.
    Zero external dependencies, fast, tamper-proof.
    Format: base64(payload).signature_hex
    """
    payload = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "picture": picture,
        "exp": int(time.time()) + expires_in,
        "iat": int(time.time()),
    }
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode('utf-8')).decode('utf-8').rstrip('=')

    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        payload_b64.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return f"{payload_b64}.{signature}"


def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validates the session token signature and expiration.
    Returns the payload dictionary if valid, or None if expired/tampered.
    """
    if not token or "." not in token:
        return None

    try:
        payload_b64, signature = token.split(".", 1)

        # Verify HMAC signature
        expected_sig = hmac.new(
            SECRET_KEY.encode('utf-8'),
            payload_b64.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            logger.warning("[Auth] Invalid session token signature.")
            return None

        # Pad base64 string if needed
        padding = len(payload_b64) % 4
        if padding > 0:
            payload_b64 += '=' * (4 - padding)

        payload_json = base64.urlsafe_b64decode(payload_b64.encode('utf-8')).decode('utf-8')
        payload = json.loads(payload_json)

        # Check expiration
        if time.time() > payload.get("exp", 0):
            logger.info("[Auth] Session token has expired.")
            return None

        return payload
    except Exception as e:
        logger.warning(f"[Auth] Failed to decode session token: {e}")
        return None
