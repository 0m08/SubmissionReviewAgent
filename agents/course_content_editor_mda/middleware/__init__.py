"""Middleware for the Managed Deep Agent build."""

from .editor_identity import editor_identity
from .present_bridge import present_to_client

__all__ = ["editor_identity", "present_to_client"]
