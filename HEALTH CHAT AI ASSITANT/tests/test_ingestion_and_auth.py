"""Tests for Document Ingestion, Approval Workflow, and Access Control Matrix."""

import unittest
from pathlib import Path
from src.access_control import UserRole, get_allowed_access_levels, is_authorized_for_level
from src.config import SecurityConfig
from src.ingestion import IngestionSecurityError, SecureDocumentIngestion
from src.vector_store import SecureVectorStore


class TestIngestionAndAuth(unittest.TestCase):

    def setUp(self):
        self.ingestion = SecureDocumentIngestion()
        self.vector_store = SecureVectorStore()

    def test_rbac_matrix(self):
        # Citizen permissions
        self.assertTrue(is_authorized_for_level(UserRole.CITIZEN, "PUBLIC"))
        self.assertFalse(is_authorized_for_level(UserRole.CITIZEN, "STAFF_ONLY"))
        self.assertFalse(is_authorized_for_level(UserRole.CITIZEN, "RESTRICTED_ADMIN"))

        # Staff permissions
        self.assertTrue(is_authorized_for_level(UserRole.STAFF, "PUBLIC"))
        self.assertTrue(is_authorized_for_level(UserRole.STAFF, "STAFF_ONLY"))
        self.assertFalse(is_authorized_for_level(UserRole.STAFF, "RESTRICTED_ADMIN"))

        # Admin permissions
        self.assertTrue(is_authorized_for_level(UserRole.ADMIN, "PUBLIC"))
        self.assertTrue(is_authorized_for_level(UserRole.ADMIN, "STAFF_ONLY"))
        self.assertTrue(is_authorized_for_level(UserRole.ADMIN, "RESTRICTED_ADMIN"))

    def test_document_approval_workflow(self):
        sample_doc = b"Official Hospital Outpatient Schedule: OPD operates from 8 AM to 2 PM."
        # Stage document
        meta = self.ingestion.stage_upload(
            filename="opd_schedule.txt",
            file_bytes=sample_doc,
            access_level=SecurityConfig.ACCESS_LEVEL_PUBLIC,
            uploaded_by="nurse_bob"
        )
        self.assertEqual(meta.status, "PENDING_APPROVAL")

        # Non-admin cannot approve
        with self.assertRaises(IngestionSecurityError):
            self.ingestion.approve_document(meta.doc_id, admin_user="citizen_dan", admin_role="CITIZEN")

        # Admin approves
        approved_meta = self.ingestion.approve_document(meta.doc_id, admin_user="dr_admin", admin_role="ADMIN")
        self.assertEqual(approved_meta.status, "APPROVED")
        self.assertEqual(approved_meta.approved_by, "dr_admin")

        # Check retrieval of approved text
        text = self.ingestion.get_document_text(meta.doc_id)
        self.assertIsNotNone(text)
        self.assertIn("OPD operates", text)


if __name__ == "__main__":
    unittest.main()
