from __future__ import annotations


class CliError(RuntimeError):
    """Recoverable command error that should be rendered by the CLI boundary."""


def format_cli_error(exc: CliError) -> str:
    message = str(exc).strip()
    if message.startswith("ERROR:"):
        return message
    return f"ERROR: {message}"


__all__ = ["CliError", "format_cli_error"]
