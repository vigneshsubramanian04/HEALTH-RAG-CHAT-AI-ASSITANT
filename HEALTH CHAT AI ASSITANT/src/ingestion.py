"""Secure Document Ingestion Layer for Hospital RAG Chatbot.

Enforces:
- File extension and magic byte signature verification
- Path traversal and malicious filename prevention
- File size and extracted text length limits
- Macro and script scanning
- Server-generated document IDs (UUID4 + SHA-256)
- Isolation: Uploads placed in data/pending/ until explicit Admin approval
- Anti-tampering: Immutability of approved documents
"""

import csv
import hashlib
import io
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import SecurityConfig
from src.guardrails import sanitize_document_text_for_rag
from src.logger import log_ingestion_event, log_security_event


# Magic byte signatures
MAGIC_SIGNATURES = {
    ".pdf": [b"%PDF-"],
    ".docx": [b"PK\x03\x04"],  # Standard ZIP signature used by DOCX
}

# Dangerous executable / macro indicators
DANGEROUS_SIGNATURES = [
    b"MZ",                     # Windows EXE / DLL
    b"\x7fELF",                # Linux ELF binary
    b"vbaProject.bin",         # MS Office VBA Macro
]


class IngestionSecurityError(Exception):
    """Raised when an ingested file violates security policies."""
    pass


@dataclass
class DocumentMetadata:
    """Audit-tracked metadata for ingested departmental documents."""
    doc_id: str
    filename: str
    content_hash: str
    access_level: str
    status: str  # "PENDING_APPROVAL", "APPROVED", "REJECTED"
    file_size_bytes: int
    char_count: int
    pages: int
    version: str
    effective_date: str
    uploaded_by: str
    approved_by: Optional[str] = None
    approval_date: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SecureDocumentIngestion:
    """Manages secure upload, content validation, and administrator approval."""

    def __init__(self):
        SecurityConfig.ensure_directories()
        self.pending_dir = SecurityConfig.PENDING_DOCS_DIR
        self.approved_dir = SecurityConfig.APPROVED_DOCS_DIR
        self.registry_file = SecurityConfig.DATA_DIR / "document_registry.json"
        self._ensure_registry()

    def _ensure_registry(self):
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_registry(self, registry: Dict[str, Dict[str, Any]]):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Deterministic prevention of path traversal and illegal characters."""
        if not filename or not isinstance(filename, str):
            raise IngestionSecurityError("Invalid filename provided.")

        # Disallow dot-dot sequences anywhere in input
        if ".." in filename:
            raise IngestionSecurityError("Path traversal pattern detected in filename.")

        # Prevent null bytes and control chars
        clean_name = filename.replace("\x00", "").strip()

        # Remove path separators (prevent ../ or ..\ traversal)
        clean_name = os.path.basename(clean_name.replace("/", os.sep).replace("\\", os.sep))

        # Disallow dot-dot sequences
        if ".." in clean_name:
            raise IngestionSecurityError("Path traversal pattern detected in filename.")

        # Whitelist safe characters: alphanumeric, dashes, underscores, dots
        safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name)
        if not safe_name or safe_name.startswith("."):
            safe_name = f"doc_{uuid.uuid4().hex[:8]}.txt"

        return safe_name

    def validate_file_bytes(self, filename: str, file_bytes: bytes) -> Tuple[str, str]:
        """Validates file extension, magic signature, file size, and macro safety."""
        # 1. Size verification
        file_size = len(file_bytes)
        if file_size == 0:
            raise IngestionSecurityError("Uploaded file is empty.")
        if file_size > SecurityConfig.MAX_UPLOAD_FILE_SIZE_BYTES:
            raise IngestionSecurityError(
                f"File size ({file_size} bytes) exceeds limit of {SecurityConfig.MAX_UPLOAD_FILE_SIZE_BYTES} bytes."
            )

        # 2. Extension verification
        safe_name = self.sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in SecurityConfig.ALLOWED_EXTENSIONS:
            raise IngestionSecurityError(
                f"Unsupported file extension '{ext}'. Approved formats: {list(SecurityConfig.ALLOWED_EXTENSIONS)}"
            )

        # 3. Executable / Macro detection
        for danger in DANGEROUS_SIGNATURES:
            if file_bytes.startswith(danger) or (b"vbaProject.bin" in file_bytes):
                raise IngestionSecurityError("Dangerous executable code or embedded macros detected in upload.")

        # 4. Magic signature verification
        if ext in MAGIC_SIGNATURES:
            signatures = MAGIC_SIGNATURES[ext]
            matched = any(file_bytes.startswith(sig) for sig in signatures)
            if not matched:
                raise IngestionSecurityError(
                    f"File signature mismatch. File claimed to be '{ext}' but signature does not match."
                )

        # Compute deterministic SHA-256 hash
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        return safe_name, content_hash

    def extract_text(self, file_bytes: bytes, ext: str) -> Tuple[str, int]:
        """Safely extracts text and counts pages. Enforces extracted character limits."""
        text_content = ""
        page_count = 1

        if ext == ".pdf":
            try:
                # Try pypdf first if available
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                page_count = len(reader.pages)
                extracted_pages = []
                for idx, page in enumerate(reader.pages):
                    ptxt = page.extract_text() or ""
                    extracted_pages.append(f"[Page {idx + 1}]\n{ptxt}")
                text_content = "\n\n".join(extracted_pages)
            except ImportError:
                # Fallback for plain text parsing of PDF streams or minimal extract
                text_content = file_bytes.decode("latin1", errors="ignore")
                # Basic stream extraction
                streams = re.findall(r"stream[\r\n]+(.*?)[\r\n]+endstream", text_content, re.DOTALL)
                text_content = "\n".join(streams) if streams else text_content
            except Exception as e:
                raise IngestionSecurityError(f"Malformed PDF file could not be parsed: {str(e)}")

        elif ext in (".txt", ".md"):
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_content = file_bytes.decode("latin1")
                except Exception as e:
                    raise IngestionSecurityError(f"Text encoding error: {str(e)}")

        elif ext == ".csv":
            try:
                decoded = file_bytes.decode("utf-8", errors="replace")
                reader = csv.reader(io.StringIO(decoded))
                rows = [", ".join(row) for row in reader]
                text_content = "\n".join(rows)
            except Exception as e:
                raise IngestionSecurityError(f"CSV parsing error: {str(e)}")

        elif ext == ".docx":
            # Basic safe XML text extraction from docx zip without executing macros
            import zipfile
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    # Check for macro components
                    if "word/vbaProject.bin" in z.namelist():
                        raise IngestionSecurityError("DOCX contains dangerous VBA macros.")
                    xml_content = z.read("word/document.xml").decode("utf-8", errors="ignore")
                    # Strip XML tags safely
                    text_content = re.sub(r"<[^>]+>", " ", xml_content)
                    text_content = re.sub(r"\s+", " ", text_content).strip()
            except zipfile.BadZipFile:
                raise IngestionSecurityError("Malformed DOCX archive.")
            except Exception as e:
                raise IngestionSecurityError(f"DOCX extraction error: {str(e)}")

        # Enforce extracted text maximum limit
        if len(text_content) > SecurityConfig.MAX_EXTRACTED_TEXT_LENGTH:
            raise IngestionSecurityError(
                f"Extracted document text ({len(text_content)} chars) exceeds limit of "
                f"{SecurityConfig.MAX_EXTRACTED_TEXT_LENGTH} characters."
            )

        # Sanitize against indirect prompt injection
        clean_text, had_injections = sanitize_document_text_for_rag(text_content)
        if had_injections:
            log_security_event(
                "DOCUMENT_INJECTION_STRIPPED",
                "ingestion",
                "ADMIN",
                "HIGH",
                {"detail": "Indirect prompt injection syntax neutralized in document text"}
            )

        return clean_text, max(page_count, 1)

    def stage_upload(
        self,
        filename: str,
        file_bytes: bytes,
        access_level: str = SecurityConfig.ACCESS_LEVEL_PUBLIC,
        uploaded_by: str = "staff_member",
        version: str = "1.0",
        effective_date: Optional[str] = None,
    ) -> DocumentMetadata:
        """Stages an uploaded document into pending directory awaiting admin review."""
        safe_name, content_hash = self.validate_file_bytes(filename, file_bytes)
        ext = Path(safe_name).suffix.lower()

        # Validate access level category
        valid_levels = {
            SecurityConfig.ACCESS_LEVEL_PUBLIC,
            SecurityConfig.ACCESS_LEVEL_STAFF,
            SecurityConfig.ACCESS_LEVEL_RESTRICTED,
        }
        if access_level not in valid_levels:
            access_level = SecurityConfig.ACCESS_LEVEL_PUBLIC

        # Extract and sanitize text
        clean_text, page_count = self.extract_text(file_bytes, ext)

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()
        eff_date = effective_date or now_str[:10]

        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=safe_name,
            content_hash=content_hash,
            access_level=access_level,
            status="PENDING_APPROVAL",
            file_size_bytes=len(file_bytes),
            char_count=len(clean_text),
            pages=page_count,
            version=version,
            effective_date=eff_date,
            uploaded_by=uploaded_by,
        )

        # Save to pending directory
        pending_file = self.pending_dir / f"{doc_id}_{safe_name}"
        with open(pending_file, "wb") as f:
            f.write(file_bytes)

        # Save extracted text
        text_file = self.pending_dir / f"{doc_id}_text.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(clean_text)

        # Update registry
        reg = self._load_registry()
        reg[doc_id] = asdict(meta)
        self._save_registry(reg)

        log_ingestion_event(doc_id, safe_name, content_hash, access_level, "STAGED_PENDING", uploaded_by)
        return meta

    def approve_document(
        self,
        doc_id: str,
        admin_user: str,
        admin_role: str = "ADMIN",
    ) -> DocumentMetadata:
        """Admin-only workflow to approve a staged document and mark it ready for indexing."""
        if admin_role.upper() != "ADMIN":
            log_security_event("UNAUTHORIZED_APPROVAL_ATTEMPT", "system", admin_role, "HIGH", {"doc_id": doc_id})
            raise IngestionSecurityError("Only authorized administrators can approve documents for indexing.")

        reg = self._load_registry()
        if doc_id not in reg:
            raise IngestionSecurityError(f"Document ID '{doc_id}' not found in registry.")

        data = reg[doc_id]
        if data["status"] == "APPROVED":
            return DocumentMetadata(**data)

        # Move files from pending to approved
        safe_name = data["filename"]
        pending_file = self.pending_dir / f"{doc_id}_{safe_name}"
        pending_text = self.pending_dir / f"{doc_id}_text.txt"

        approved_file = self.approved_dir / f"{doc_id}_{safe_name}"
        approved_text = self.approved_dir / f"{doc_id}_text.txt"

        if pending_file.exists():
            pending_file.replace(approved_file)
        if pending_text.exists():
            pending_text.replace(approved_text)

        now_str = datetime.now(timezone.utc).isoformat()
        data["status"] = "APPROVED"
        data["approved_by"] = admin_user
        data["approval_date"] = now_str

        reg[doc_id] = data
        self._save_registry(reg)

        log_ingestion_event(
            doc_id, safe_name, data["content_hash"], data["access_level"], "APPROVED", admin_user
        )
        return DocumentMetadata(**data)

    def reject_document(self, doc_id: str, admin_user: str, admin_role: str = "ADMIN", reason: str = "Policy violation") -> bool:
        """Reject and purge an unapproved or malicious upload."""
        if admin_role.upper() != "ADMIN":
            raise IngestionSecurityError("Only administrators can reject documents.")

        reg = self._load_registry()
        if doc_id in reg:
            data = reg[doc_id]
            data["status"] = "REJECTED"
            reg[doc_id] = data
            self._save_registry(reg)

            # Clean pending files
            for p in self.pending_dir.glob(f"{doc_id}*"):
                try:
                    p.unlink()
                except Exception:
                    pass

            log_ingestion_event(doc_id, data["filename"], data["content_hash"], data["access_level"], f"REJECTED: {reason}", admin_user)
            return True
        return False

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Retrieve approved sanitized document text for vector indexing."""
        reg = self._load_registry()
        if doc_id not in reg or reg[doc_id]["status"] != "APPROVED":
            return None

        approved_text = self.approved_dir / f"{doc_id}_text.txt"
        if approved_text.exists():
            with open(approved_text, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def list_documents(self, status: Optional[str] = None) -> List[DocumentMetadata]:
        """List registered documents filtered by status."""
        reg = self._load_registry()
        docs = [DocumentMetadata(**item) for item in reg.values()]
        if status:
            docs = [d for d in docs if d.status == status.upper()]
        return docs


document_ingestion_service = SecureDocumentIngestion()
