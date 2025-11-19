"""
Paraphraser Agent - Multi-Agent Orchestrator

Transforms monotonous technical text into clear, conversational language
that sounds like an experienced tradesperson.
"""

from .orchestrator import run_paraphraser, create_paraphraser_orchestrator
from .state import ParaphraserState

__all__ = [
    'run_paraphraser',
    'create_paraphraser_orchestrator',
    'ParaphraserState',
]
