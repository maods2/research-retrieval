import os
from pathlib import Path
from typing import Optional

def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment.
    
    Tries to load from .env file first using python-dotenv if available,
    then falls back to environment variables.
    
    Returns None if not set (needed for models that don't require HF auth).
    """
    # Try to load from .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        # Find the project root (where .env should be)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Go up from src/utils/ to root
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        # python-dotenv not available, will use system env vars only
        pass
    
    token = os.getenv("HF_TOKEN")
    # Return None if empty string
    return token if token else None