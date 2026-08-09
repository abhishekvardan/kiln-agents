class KilnError(Exception):
    """Raised for any kiln-specific failure (provider errors, schema validation exhaustion,
    orchestration graph errors, MCP connection failures, ...)."""
