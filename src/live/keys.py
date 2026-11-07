"""Check that D's API keys in .env are set and accepted.

Usage:
    python -m src.live.keys

Makes one cheap read-only call to each service. Never prints the keys.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.common.contracts import ROOT


def check_gemini(key: str) -> str:
    from google import genai
    models = list(genai.Client(api_key=key).models.list())
    return f"ok ({len(models)} models visible)"


def check_elevenlabs(key: str) -> str:
    from elevenlabs.client import ElevenLabs
    models = ElevenLabs(api_key=key).models.list()
    return f"ok ({len(models)} models visible)"


CHECKS = {"GEMINI_API_KEY": check_gemini, "ELEVENLABS_API_KEY": check_elevenlabs}


def main(argv: list[str] | None = None) -> bool:
    load_dotenv(ROOT / ".env")
    all_ok = True
    for name, check in CHECKS.items():
        key = os.getenv(name, "").strip()
        if not key:
            print(f"{name}: missing (add it to .env)")
            all_ok = False
            continue
        try:
            print(f"{name}: {check(key)}")
        except Exception as exc:  # report any auth/network failure, keep going
            print(f"{name}: FAILED ({type(exc).__name__}: {str(exc)[:200]})")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
