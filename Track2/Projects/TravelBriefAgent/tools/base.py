import os
import re
import time
import logging
import functools
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets when inside a running app, otherwise fall back to env vars."""
    try:
        import streamlit as st
        if st.runtime.exists():
            return st.secrets.get(key) or os.getenv(key, default)
    except Exception:
        pass
    return os.getenv(key, default)


def sanitize_input(value: str, max_len: int = 200) -> str:
    """Strip shell metacharacters and control chars; truncate to prevent injection via tool inputs."""
    if not isinstance(value, str):
        return str(value)[:max_len]
    # Allowlist approach: strip shell metacharacters AND newlines/carriage-returns
    cleaned = re.sub(r'[<>{}\[\]\\`|;$\n\r]', '', value)
    return cleaned.strip()[:max_len]


def retry(max_attempts: int = 3, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator: retry on specified exceptions with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts - 1:
                        raise RuntimeError(
                            f"[failed after {max_attempts} retries] {exc}"
                        ) from exc
                    wait = backoff ** attempt
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                        func.__name__, attempt + 1, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


class Tool(ABC):
    name = ""
    description = ""
    TIMEOUT = 10  # default HTTP timeout seconds

    @abstractmethod
    def run(self, **kwargs):
        pass
