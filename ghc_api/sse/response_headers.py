"""Allowlisted upstream metadata whose meaning survives Responses passthrough."""

from typing import Dict


REASONING_INCLUDED_HEADER = "x-reasoning-included"


def reasoning_accounting_headers(response) -> Dict[str, str]:
    """Forward only a real, positive declaration from this upstream response.

    Codex treats the header's *presence* as true, even for the text ``false``.
    Omit false, empty, missing and unknown values rather than accidentally
    telling the client that unreported reasoning is already counted. Never
    forward authentication, cookies or arbitrary upstream headers here.
    """
    headers = getattr(response, "headers", None) or {}
    for name, value in headers.items():
        if (isinstance(name, str) and name.lower() == REASONING_INCLUDED_HEADER
                and isinstance(value, str) and value.strip().lower() == "true"):
            return {REASONING_INCLUDED_HEADER: value.strip()}
    return {}
