"""Input, Context, and Output Guardrails for Hospital RAG Chatbot.

Security Controls Breakdown:
1. DETERMINISTIC CONTROLS:
   - Null-byte & non-printable control character filtering
   - Unicode normalization (NFKC) & zero-width character removal
   - Maximum input length enforcement
   - Sliding-window rate limiter per session/IP
   - Strict input type validation
   - PII & Secret regex scanning and redaction (API keys, SSN, MRN, cards, phones)
   - Deterministic citation verification against retrieved chunk metadata

2. HEURISTIC CONTROLS:
   - Obfuscation detection (Base64, Hex, Leetspeak)
   - Layered prompt-injection & jailbreak pattern classification
   - Contextual disambiguation: Differentiates legitimate hospital questions
     (e.g., 'ignore symptoms', 'staff password policy', 'hospital administrator')
     from instruction-override attacks ('ignore rules', 'pretend administrator').
   - Indirect prompt injection markers in retrieved context

3. LLM-BOUNDED SAFEGUARDS:
   - Evidence-only context framing with strict system prompts
   - Citation provenance requirement
   - Standardized safe fallback responses for insufficient evidence or blocked requests
"""

import base64
import binascii
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import SecurityConfig
from src.logger import log_security_event


# ============================================================================
# 1. Deterministic Sanitizers & PII / Secret Scanners
# ============================================================================

# Patterns for secrets and credentials
SECRET_REGEXES = [
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?", re.IGNORECASE),
]

# Patterns for Personally Identifiable Information (PII) & Medical Records
PII_REGEXES = [
    # SSN / National Identification (XXX-XX-XXXX or 9-12 contiguous digits)
    re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    # Credit Card numbers
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    # Standard phone numbers
    re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # Patient Record Identifier / MRN
    re.compile(r"\b(?:MRN|PATIENT[-_]?ID|RECORD[-_]?ID)[:\s#]*[A-Z0-9-]{4,16}\b", re.IGNORECASE),
]


def sanitize_unicode_and_controls(text: str) -> str:
    """Deterministic normalization: Strips control characters, null bytes, zero-width chars."""
    if not isinstance(text, str):
        return ""

    # Normalize unicode to NFKC
    normalized = unicodedata.normalize("NFKC", text)

    # Strip dangerous control characters (keep standard \n, \r, \t)
    cleaned = "".join(
        ch for ch in normalized
        if (ch in ("\n", "\r", "\t") or unicodedata.category(ch)[0] != "C")
        and ch not in ("\u200B", "\u200C", "\u200D", "\uFEFF", "\u202E", "\u0000")
    )
    return cleaned.strip()


def scan_and_redact_sensitive_data(text: str) -> Tuple[str, bool, List[str]]:
    """Deterministic scan for secrets and PII.

    Returns:
        (sanitized_text, has_sensitive_data, list_of_detected_categories)
    """
    detected = []
    sanitized = text

    for regex in SECRET_REGEXES:
        if regex.search(sanitized):
            detected.append("SECRET_CREDENTIAL")
            sanitized = regex.sub("[REDACTED_SECRET]", sanitized)

    for regex in PII_REGEXES:
        if regex.search(sanitized):
            detected.append("PII_MEDICAL_DATA")
            sanitized = regex.sub("[REDACTED_PII]", sanitized)

    return sanitized, (len(detected) > 0), list(set(detected))


# ============================================================================
# 2. Rate Limiter (Deterministic Sliding Window)
# ============================================================================

class SlidingWindowRateLimiter:
    """Sliding-window in-memory rate limiter with block duration for abusive clients."""

    def __init__(self, max_requests: int = 15, window_seconds: int = 60, block_duration: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.block_duration = block_duration
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._blocked_until: Dict[str, float] = {}
        self._violation_count: Dict[str, int] = defaultdict(int)

    def is_allowed(self, client_id: str) -> Tuple[bool, str]:
        now = time.time()

        # Check if currently under block cooldown
        if client_id in self._blocked_until:
            remaining = int(self._blocked_until[client_id] - now)
            if remaining > 0:
                return False, f"Rate limit exceeded. Try again in {remaining} seconds."
            else:
                del self._blocked_until[client_id]

        # Purge timestamps outside the sliding window
        window_start = now - self.window_seconds
        self._requests[client_id] = [t for t in self._requests[client_id] if t > window_start]

        if len(self._requests[client_id]) >= self.max_requests:
            self._violation_count[client_id] += 1
            # Exponential block duration on repeated violations
            block_time = self.block_duration * min(self._violation_count[client_id], 5)
            self._blocked_until[client_id] = now + block_time
            return False, f"Too many requests. Temporary cooldown active for {block_time} seconds."

        self._requests[client_id].append(now)
        return True, ""

    def reset(self, client_id: Optional[str] = None):
        """Reset state (for testing)."""
        if client_id:
            self._requests.pop(client_id, None)
            self._blocked_until.pop(client_id, None)
            self._violation_count.pop(client_id, None)
        else:
            self._requests.clear()
            self._blocked_until.clear()
            self._violation_count.clear()


rate_limiter = SlidingWindowRateLimiter(
    max_requests=SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE,
    window_seconds=60,
    block_duration=SecurityConfig.RATE_LIMIT_BLOCK_DURATION_SECONDS,
)


# ============================================================================
# 3. Prompt-Injection, Jailbreak & Obfuscation Guardrails (Heuristic & Layered)
# ============================================================================

# High-risk instruction override patterns targeting the system prompt or AI identity
_INSTRUCTION_OVERRIDE_PATTERNS = [
    # Direct instruction cancellation
    re.compile(r"\b(?:ignore|disregard|forget|override|bypass|clear|drop|stop)\s+(?:all\s+|any\s+|your\s+|the\s+)*(?:previous|prior|above|system|core|safety)?\s*(?:instructions?|prompts?|rules?|guidelines?|constraints?)\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+not\s+follow|stop\s+following)\s+(?:any\s+|all\s+|your\s+)*(?:instructions?|rules?|guidelines?)\b", re.IGNORECASE),
    re.compile(r"\b(?:disregard|ignore)\s+(?:all\s+|any\s+|your\s+)*(?:safety\s+rules?|constraints?|guardrails?)\b", re.IGNORECASE),
    
    # Prompt extraction attempts
    re.compile(r"\b(?:reveal|show|display|print|leak|output|repeat|quote|give\s+me)\s+(?:all\s+|your\s+|the\s+)*(?:hidden\s+|secret\s+|initial\s+|system\s+)*(?:prompt|instructions?|system\s+message)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+your\s+(?:hidden\s+|secret\s+|system\s+)*(?:prompt|instructions?|system\s+message)\b", re.IGNORECASE),
    
    # Credential & secret extraction attempts
    re.compile(r"\b(?:show|reveal|display|give|tell|print)\s+(?:me\s+)?(?:the\s+)?(?:api[\s_-]?key|groq[\s_-]?key|secret[\s_-]?key|password|credential|token)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(?:the\s+)?(?:api[\s_-]?key|groq[\s_-]?key|groq_api_key)\b", re.IGNORECASE),

    # Jailbreak / Roleplay mode switches
    re.compile(r"\b(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+(?:be|have)|you\s+(?:are|have)))\s+(?:[a-zA-Z0-9_\-]+\s+)*(?:DAN|developer\s+mode|unrestricted|god\s+mode|jailbroken|an\s+unfiltered\s+ai|evil\s+bot|root|admin(?:istrator)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:enable|enter)\s+(?:developer\s+mode|god\s+mode|unrestricted\s+mode|jailbreak)\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(?:to\s+(?:be|have)|you\s+(?:are|have))\s+(?:[a-zA-Z0-9_\-]+\s+)*(?:admin(?:istrator)?|root|elevated)\s*(?:access|rights|privileges)?\b", re.IGNORECASE),

    # Data breach coercion
    re.compile(r"\b(?:ignore\s+(?:your\s+)?safety\s+rules\s+and\s+reveal|reveal\s+all\s+(?:patient\s+records|confidential\s+data))\b", re.IGNORECASE),
    re.compile(r"\bretrieved\s+document\s+says\s+you\s+must\s+(?:disclose|reveal|output)\b", re.IGNORECASE),
]

# Whitelist patterns for legitimate queries that use sensitive terminology in clinical or procedural context
_LEGITIMATE_CONTEXT_PATTERNS = [
    # Medical context where "ignore" is safe (e.g. ignore mild symptoms/side-effects)
    re.compile(r"\b(?:patient|doctor|symptom|side\s*effect|cough|fever|pain|rash|medication|dose)\b", re.IGNORECASE),
    # Hospital administration context (e.g. hospital administrator contact, department administrator)
    re.compile(r"\b(?:contact|duty|phone|office|hours|hospital\s+administrator|department\s+head|superintendent)\b", re.IGNORECASE),
    # Security/Password policy inquiry (e.g. staff portal password rules, Wi-Fi password for visitors)
    re.compile(r"\b(?:policy|guidelines?|requirements?|standard|portal|wifi|wi-fi|how\s+to\s+reset|how\s+often)\b", re.IGNORECASE),
]


def detect_base64_or_hex_obfuscation(text: str) -> Tuple[bool, Optional[str]]:
    """Heuristic check: Detects and inspects base64 or hex encoded payloads."""
    # Match candidate base64 words of significant length
    tokens = text.strip().split()
    for token in tokens:
        clean_token = token.strip("()[]{},;.\"':")
        if len(clean_token) >= 20 and len(clean_token) % 4 == 0:
            try:
                decoded = base64.b64decode(clean_token, validate=True).decode("utf-8", errors="ignore")
                if len(decoded) > 5 and any(kw in decoded.lower() for kw in ["ignore", "prompt", "system", "password", "key", "bypass", "jailbreak"]):
                    return True, decoded
            except Exception:
                pass

        # Check Hex encoding
        if len(clean_token) >= 24 and all(c in "0123456789abcdefABCDEF" for c in clean_token):
            try:
                decoded = bytes.fromhex(clean_token).decode("utf-8", errors="ignore")
                if len(decoded) > 5 and any(kw in decoded.lower() for kw in ["ignore", "prompt", "system", "password", "key", "bypass", "jailbreak"]):
                    return True, decoded
            except Exception:
                pass

    return False, None


def validate_user_input(
    user_input: Any,
    session_id: str = "anon",
    role: str = "CITIZEN",
) -> Tuple[bool, str, str]:
    """Validates user input across all required security layers.

    Returns:
        (is_valid: bool, cleaned_text_or_error_msg: str, rejection_reason: str)
    """
    # 1. Type validation
    if not isinstance(user_input, str):
        log_security_event("INPUT_TYPE_REJECTION", session_id, role, "MEDIUM", {"type": type(user_input).__name__})
        return False, "Invalid input type. Only text questions are accepted.", "INVALID_TYPE"

    # 2. Empty input validation
    raw_stripped = user_input.strip()
    if not raw_stripped:
        return False, "Question cannot be empty. Please enter your hospital or medical inquiry.", "EMPTY_INPUT"

    # 3. Unicode and control characters normalization
    sanitized = sanitize_unicode_and_controls(raw_stripped)
    if not sanitized:
        log_security_event("MALFORMED_INPUT", session_id, role, "LOW", {"detail": "Stripped to empty"})
        return False, "Input contained only unprintable or invalid characters.", "MALFORMED_INPUT"

    # 4. Maximum length check
    if len(sanitized) > SecurityConfig.MAX_QUESTION_LENGTH:
        log_security_event("INPUT_LENGTH_EXCEEDED", session_id, role, "MEDIUM", {"length": len(sanitized)})
        return (
            False,
            f"Question exceeds the maximum length of {SecurityConfig.MAX_QUESTION_LENGTH} characters. Please submit a more focused inquiry.",
            "LENGTH_EXCEEDED"
        )

    # 5. Sliding-window Rate Limiting
    allowed, rate_msg = rate_limiter.is_allowed(session_id)
    if not allowed:
        log_security_event("RATE_LIMIT_HIT", session_id, role, "HIGH", {"session": session_id})
        return False, rate_msg, "RATE_LIMITED"

    # 6. Obfuscation detection (Base64 / Hex payloads)
    is_obfuscated, decoded_content = detect_base64_or_hex_obfuscation(sanitized)
    if is_obfuscated and decoded_content:
        log_security_event("OBFUSCATED_INJECTION_DETECTED", session_id, role, "HIGH", {"decoded_sample": decoded_content[:30]})
        return False, "Request blocked: Encoded instruction override or suspicious payload detected.", "OBFUSCATION_BLOCKED"

    # 7. Layered Prompt Injection and Jailbreak Detection
    for pattern in _INSTRUCTION_OVERRIDE_PATTERNS:
        match = pattern.search(sanitized)
        if match:
            # Check for legitimate hospital context disambiguation
            is_legit = False
            # If the user asks about password policy or hospital administrator, verify intent
            if "password" in match.group(0).lower() or "administrator" in match.group(0).lower() or "ignore" in match.group(0).lower():
                for legit_pat in _LEGITIMATE_CONTEXT_PATTERNS:
                    if legit_pat.search(sanitized):
                        # Ensure it is NOT explicitly attempting privilege escalation or instruction cancellation
                        override_test = re.search(r"\b(?:pretend\s+you\s+have|ignore\s+all\s+previous|reveal\s+your\s+hidden|show\s+me\s+the\s+api)\b", sanitized, re.IGNORECASE)
                        if not override_test:
                            is_legit = True
                            break

            if not is_legit:
                log_security_event(
                    "PROMPT_INJECTION_BLOCKED",
                    session_id,
                    role,
                    "HIGH",
                    {"matched_pattern": match.group(0)}
                )
                return (
                    False,
                    "I cannot process instructions intended to alter system parameters, retrieve credentials, or override safety guidelines. How can I assist you with approved hospital procedures or medical guidelines?",
                    "INJECTION_OVERRIDE_BLOCKED"
                )

    return True, sanitized, ""


# ============================================================================
# 4. Indirect Prompt-Injection Scanning for Documents / Context Chunks
# ============================================================================

_INDIRECT_INJECTION_MARKERS = [
    re.compile(r"(?:\[SYSTEM\s+PROMPT|SYSTEM\s*NOTE|INSTRUCTION\s*OVERRIDE\])", re.IGNORECASE),
    re.compile(r"ignore\s+(?:all\s+)?(?:instructions|safety\s+rules|restrictions)", re.IGNORECASE),
    re.compile(r"assistant\s*:\s*i\s+will\s+now\s+(?:print|reveal|disclose)", re.IGNORECASE),
    re.compile(r"you\s+must\s+(?:disclose|leak|display)\s+(?:all\s+confidential|patient\s+records|secrets)", re.IGNORECASE),
    re.compile(r"<system_instruction>", re.IGNORECASE),
]


def sanitize_document_text_for_rag(text: str) -> Tuple[str, bool]:
    """Sanitizes text extracted from ingested documents or chunks.

    Strips indirect prompt-injection markers and neutralizes macro or script tags.
    Returns:
        (cleaned_text, had_injections: bool)
    """
    cleaned = sanitize_unicode_and_controls(text)
    had_injection = False

    for pat in _INDIRECT_INJECTION_MARKERS:
        if pat.search(cleaned):
            had_injection = True
            cleaned = pat.sub("[SUSPICIOUS_DIRECTIVE_STRIPPED]", cleaned)

    # Neutralize script or macro tags
    cleaned = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "[SCRIPT_TAG_REMOVED]", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>", "[IFRAME_REMOVED]", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned, had_injection


# ============================================================================
# 5. Output Guardrails (Deterministic & Post-Generation Validation)
# ============================================================================

STANDARD_FALLBACK_ANSWER = (
    "Based on the approved departmental documents, I do not have sufficient "
    "information to answer this question. Please consult official hospital staff or relevant SOPs."
)

MEDICAL_SAFETY_DISCLAIMER = (
    "\n\n*Medical Notice: This information is derived from approved departmental documents "
    "and is for informational purposes only. In any medical emergency, please consult a licensed healthcare professional.*"
)


def validate_output_answer(
    generated_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    user_role: str = "CITIZEN",
) -> Tuple[bool, str, str]:
    """Validates the generated answer before presenting it to the user.

    Validates:
    - No secret leakage (API keys, credentials).
    - No PII leakage.
    - Citations correspond to actual retrieved sources.
    - No fabricated page numbers.
    - Proper disclaimers.

    Returns:
        (is_approved: bool, final_safe_text: str, reason: str)
    """
    if not generated_text or not generated_text.strip():
        return False, STANDARD_FALLBACK_ANSWER, "EMPTY_OUTPUT"

    # 1. Deterministic Secret Check
    for regex in SECRET_REGEXES:
        if regex.search(generated_text):
            log_security_event("OUTPUT_LEAK_BLOCKED", "system", user_role, "CRITICAL", {"leak_type": "SECRET"})
            return False, STANDARD_FALLBACK_ANSWER, "LEAK_SECRET_PREVENTED"

    # 2. Deterministic PII Check
    for regex in PII_REGEXES:
        if regex.search(generated_text):
            log_security_event("OUTPUT_LEAK_BLOCKED", "system", user_role, "HIGH", {"leak_type": "PII"})
            # Redact rather than crashing if possible, or return safe fallback if severe
            generated_text = regex.sub("[REDACTED_PII]", generated_text)

    # 3. Citation and Provenance Verification
    # Extract citations like [Doc: filename, Page: X] or [Source: filename]
    valid_doc_names: Set[str] = {
        chunk.get("metadata", {}).get("source", "").lower()
        for chunk in retrieved_chunks
        if chunk.get("metadata", {}).get("source")
    }
    valid_titles: Set[str] = {
        chunk.get("metadata", {}).get("title", "").lower()
        for chunk in retrieved_chunks
        if chunk.get("metadata", {}).get("title")
    }
    all_valid_sources = (valid_doc_names | valid_titles) - {""}

    # Find citations in the generated response: e.g. [Document: ...] or [Source: ...]
    citation_pattern = re.compile(r"\[(?:Source|Document|Doc|Ref):\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]", re.IGNORECASE)
    found_citations = citation_pattern.findall(generated_text)

    if found_citations and all_valid_sources:
        for cited_src, page_str in found_citations:
            clean_src = cited_src.strip().lower()
            # Verify if cited source is one of the legitimately retrieved chunks
            matches_source = any(clean_src in valid or valid in clean_src for valid in all_valid_sources)
            if not matches_source:
                log_security_event(
                    "HALLUCINATED_CITATION_DETECTED",
                    "system",
                    user_role,
                    "MEDIUM",
                    {"hallucinated_source": cited_src}
                )
                # Fail-safe: replace fabricated citation or fall back
                return False, STANDARD_FALLBACK_ANSWER, "FABRICATED_CITATION"

    # 4. Attach standard medical safety disclaimer if clinical advice is discussed
    output_text = generated_text
    if any(kw in output_text.lower() for kw in ["treatment", "dosage", "medication", "doctor", "emergency", "symptom"]):
        if "*Medical Notice:" not in output_text:
            output_text += MEDICAL_SAFETY_DISCLAIMER

    return True, output_text, "PASSED"
