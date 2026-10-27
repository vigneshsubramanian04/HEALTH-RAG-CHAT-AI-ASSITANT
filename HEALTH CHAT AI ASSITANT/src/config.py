"""Security Configuration for Hospital RAG Chatbot.

All credentials and tunable security limits are centralized here.
Zero credentials or sensitive secrets are hardcoded.
"""

import os
from pathlib import Path
from typing import Optional

# Optional dotenv loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Minimal fallback loader if python-dotenv is not installed
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


class SecurityConfig:
    """Central configuration for application security, models, and boundaries."""

    # Base Directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    APPROVED_DOCS_DIR = DATA_DIR / "approved"
    PENDING_DOCS_DIR = DATA_DIR / "pending"
    CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"
    LOG_FILE = LOGS_DIR / "security_audit.log"

    # External LLM Settings (Groq)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20.0"))
    GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "2"))
    GROQ_MAX_OUTPUT_TOKENS: int = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1024"))

    # Local Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Guardrails Limits
    MAX_QUESTION_LENGTH: int = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "15"))
    RATE_LIMIT_BLOCK_DURATION_SECONDS: int = int(os.getenv("RATE_LIMIT_BLOCK_DURATION_SECONDS", "60"))
    
    # Secure Ingestion Limits
    MAX_UPLOAD_FILE_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10MB
    MAX_EXTRACTED_TEXT_LENGTH: int = int(os.getenv("MAX_EXTRACTED_TEXT_LENGTH", "500000"))
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}
    
    # Document Access Levels
    ACCESS_LEVEL_PUBLIC = "PUBLIC"
    ACCESS_LEVEL_STAFF = "STAFF_ONLY"
    ACCESS_LEVEL_RESTRICTED = "RESTRICTED_ADMIN"

    # Departmental Policies
    ALLOW_EXTERNAL_LLM_PROCESSING: bool = os.getenv("ALLOW_EXTERNAL_LLM_PROCESSING", "true").lower() in ("true", "1", "yes")
    REQUIRE_ADMIN_APPROVAL: bool = os.getenv("REQUIRE_ADMIN_APPROVAL", "true").lower() in ("true", "1", "yes")

    @classmethod
    def ensure_directories(cls):
        """Ensure all required data and log directories exist safely."""
        cls.APPROVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PENDING_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def has_groq_key(cls) -> bool:
        """Check if Groq API key is present without disclosing it."""
        key = cls.GROQ_API_KEY
        return bool(key and key.strip() and not key.startswith("gsk_your_groq_api_key"))


# Auto-create directories on import
SecurityConfig.ensure_directories()
