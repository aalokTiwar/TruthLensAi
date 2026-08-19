"""
Project configuration loader.

Reads all environment variables from .env and makes them available
through a single Settings object.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Central configuration class."""

    # Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Backend
    BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", 5))

    # Agent
    MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", 2))


# Global settings object
settings = Settings()