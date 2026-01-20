import os

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")
    return token