"""Role-Based Access Control (RBAC) and Authorization Engine.

Enforces server-side authorization before retrieval from ChromaDB.
Eliminates UI-level bypasses and prevents unauthorized document leakage.
"""

from enum import Enum
from typing import Any, Dict, List, Set

from src.config import SecurityConfig
from src.logger import log_security_event


class UserRole(str, Enum):
    CITIZEN = "CITIZEN"
    STAFF = "STAFF"
    ADMIN = "ADMIN"


class AccessLevel(str, Enum):
    PUBLIC = "PUBLIC"
    STAFF_ONLY = "STAFF_ONLY"
    RESTRICTED_ADMIN = "RESTRICTED_ADMIN"


# Role-to-Access-Level Mapping Matrix
ROLE_PERMISSIONS: Dict[UserRole, Set[AccessLevel]] = {
    UserRole.CITIZEN: {
        AccessLevel.PUBLIC,
    },
    UserRole.STAFF: {
        AccessLevel.PUBLIC,
        AccessLevel.STAFF_ONLY,
    },
    UserRole.ADMIN: {
        AccessLevel.PUBLIC,
        AccessLevel.STAFF_ONLY,
        AccessLevel.RESTRICTED_ADMIN,
    },
}


def normalize_role(role_str: str) -> UserRole:
    """Safely normalizes role string to UserRole enum. Defaults to least privilege (CITIZEN)."""
    if not isinstance(role_str, str):
        return UserRole.CITIZEN

    cleaned = role_str.strip().upper()
    try:
        return UserRole(cleaned)
    except ValueError:
        return UserRole.CITIZEN


def get_allowed_access_levels(role: UserRole | str) -> List[str]:
    """Returns list of access levels an authenticated role is authorized to read."""
    if isinstance(role, str):
        role = normalize_role(role)
    allowed = ROLE_PERMISSIONS.get(role, {AccessLevel.PUBLIC})
    return [lvl.value for lvl in allowed]


def is_authorized_for_level(user_role: UserRole | str, required_level: str) -> bool:
    """Deterministic server-side check for document access."""
    if isinstance(user_role, str):
        user_role = normalize_role(user_role)

    allowed_levels = ROLE_PERMISSIONS.get(user_role, {AccessLevel.PUBLIC})
    try:
        req_enum = AccessLevel(required_level.upper())
        return req_enum in allowed_levels
    except ValueError:
        return False


def can_approve_documents(user_role: UserRole | str) -> bool:
    """Only ADMIN role can approve or reject departmental documents."""
    if isinstance(user_role, str):
        user_role = normalize_role(user_role)
    return user_role == UserRole.ADMIN


def can_manage_vector_store(user_role: UserRole | str) -> bool:
    """Only ADMIN role can rebuild or clear the vector store index."""
    if isinstance(user_role, str):
        user_role = normalize_role(user_role)
    return user_role == UserRole.ADMIN


def can_view_audit_logs(user_role: UserRole | str) -> bool:
    """Only ADMIN role can inspect security audit logs."""
    if isinstance(user_role, str):
        user_role = normalize_role(user_role)
    return user_role == UserRole.ADMIN


def build_chroma_access_filter(user_role: UserRole | str) -> Dict[str, Any]:
    """Constructs deterministic ChromaDB metadata filter based on user role.

    Guarantees that ChromaDB query results will strictly omit unauthorized documents,
    even if an attacker asks for a restricted document explicitly by filename.
    """
    allowed = get_allowed_access_levels(user_role)

    if len(allowed) == 1:
        return {"access_level": allowed[0]}
    else:
        return {"access_level": {"$in": allowed}}
