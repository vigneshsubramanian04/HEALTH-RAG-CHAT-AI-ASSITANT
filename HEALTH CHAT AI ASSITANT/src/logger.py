"""Privacy-conscious, injection-resistant logging and audit system.

Enforces:
- Sanitization of PII, tokens, and API keys before writing to logs.
- Log injection prevention (escapes newline and control characters).
- Structured audit event tracking (session ID, role, event category, result).
- Zero retention of full prompts, patient records, or secret credentials.
"""

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from src.config import SecurityConfig

# Sensitive patterns to redact from logs
_SECRET_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}", re.IGNORECASE),  # Groq API keys
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?", re.IGNORECASE),
]

_PII_PATTERNS = [
    # SSN / Aadhaar / National ID pattern (9-12 digits with separators)
    re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    # Credit Card
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    # Email
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    # Phone numbers
    re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # Patient Record / MRN (Medical Record Number like MRN-123456)
    re.compile(r"\b(?:MRN|PATIENT[-_]?ID)[:\s#]*[A-Z0-9-]{4,12}\b", re.IGNORECASE),
]


def sanitize_log_message(msg: str) -> str:
    """Sanitize log text to prevent Log Injection, PII disclosure, and Secret leakage.

    1. Neutralizes CRLF / newlines to prevent log forging.
    2. Redacts API keys and secret patterns.
    3. Redacts identifiable personal information (PII).
    """
    if not isinstance(msg, str):
        msg = str(msg)

    # 1. Neutralize Log Injection (newlines & carriage returns)
    clean_msg = msg.replace("\r", " ").replace("\n", " [NL] ")

    # 2. Redact Secrets
    for pat in _SECRET_PATTERNS:
        clean_msg = pat.sub("[REDACTED_SECRET]", clean_msg)

    # 3. Redact PII
    for pat in _PII_PATTERNS:
        clean_msg = pat.sub("[REDACTED_PII]", clean_msg)

    return clean_msg


class SanitizingFilter(logging.Filter):
    """Custom logging filter that automatically cleans all record messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_log_message(record.msg)
        if record.args:
            cleaned_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    cleaned_args.append(sanitize_log_message(arg))
                else:
                    cleaned_args.append(arg)
            record.args = tuple(cleaned_args)
        return True


def setup_logger(name: str = "hospital_rag_security") -> logging.Logger:
    """Configures and returns the audit logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        SecurityConfig.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        sanitizer = SanitizingFilter()

        # Bounded Rotating File Handler (Max 5MB per file, max 5 backup files)
        file_handler = RotatingFileHandler(
            filename=str(SecurityConfig.LOG_FILE),
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sanitizer)
        logger.addHandler(file_handler)

        # Stream Handler for console / dev output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sanitizer)
        logger.addHandler(console_handler)

    return logger


audit_logger = setup_logger()


def log_security_event(
    event_category: str,
    session_id: str,
    role: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log security events with bounded minimal context."""
    timestamp = datetime.now(timezone.utc).isoformat()
    clean_details = {}
    if details:
        for k, v in details.items():
            clean_details[k] = sanitize_log_message(str(v))

    msg = (
        f"SECURITY_EVENT category={event_category} session={session_id} "
        f"role={role} severity={severity} details={clean_details} timestamp={timestamp}"
    )
    if severity.upper() in ("HIGH", "CRITICAL"):
        audit_logger.warning(msg)
    else:
        audit_logger.info(msg)


def log_operational_event(
    event_type: str,
    session_id: str,
    role: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log standard operational workflow status."""
    meta_str = ""
    if metadata:
        clean_meta = {k: sanitize_log_message(str(v)) for k, v in metadata.items()}
        meta_str = f" meta={clean_meta}"
    audit_logger.info(
        f"OPERATIONAL_EVENT type={event_type} session={session_id} role={role} status={status}{meta_str}"
    )


def log_ingestion_event(
    document_id: str,
    filename: str,
    file_hash: str,
    access_level: str,
    status: str,
    admin_user: Optional[str] = None,
) -> None:
    """Log document ingestion lifecycle event."""
    audit_logger.info(
        f"INGESTION_EVENT doc_id={document_id} filename={filename} "
        f"hash={file_hash[:16]}... access_level={access_level} status={status} admin={admin_user or 'system'}"
    )


def log_api_error(
    service: str,
    error_category: str,
    status_code: Optional[int] = None,
    session_id: Optional[str] = None,
) -> None:
    """Log external API errors safely without exposing prompt context or credentials."""
    audit_logger.error(
        f"API_ERROR service={service} category={error_category} "
        f"status_code={status_code or 'N/A'} session={session_id or 'anonymous'}"
    )
