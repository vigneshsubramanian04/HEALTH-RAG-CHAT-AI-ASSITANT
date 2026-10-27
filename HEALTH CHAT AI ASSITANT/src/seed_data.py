"""Seed initial approved departmental documents across all 3 RBAC tiers."""

from src.access_control import UserRole
from src.config import SecurityConfig
from src.ingestion import document_ingestion_service
from src.vector_store import vector_store_service

SAMPLE_DOCS = [
    {
        "filename": "visiting_hours_and_patient_guide.txt",
        "access_level": SecurityConfig.ACCESS_LEVEL_PUBLIC,
        "content": (
            "Hospital General Visiting Hours and Patient Guidelines\n"
            "Effective Date: 2026-01-01 | Version: 2.1\n\n"
            "1. General Ward Visiting Hours: Monday to Sunday, 10:00 AM to 12:00 PM and 4:00 PM to 7:00 PM.\n"
            "2. Intensive Care Unit (ICU): Restricted to immediate family members only. Visiting window: 5:00 PM to 6:00 PM.\n"
            "3. Outpatient Department (OPD): Operates Monday to Saturday from 8:00 AM to 2:00 PM.\n"
            "4. Emergency Services: Open 24/7 with immediate triage triage at Entrance B.\n"
            "5. Patient Code of Conduct: Quiet hours are observed from 9:00 PM to 6:00 AM. Cell phone usage should be on silent.\n"
            "6. Pharmacy: 24-hour outpatient pharmacy is located adjacent to the Main Reception."
        ),
        "uploaded_by": "reception_staff",
    },
    {
        "filename": "clinical_triage_sop.txt",
        "access_level": SecurityConfig.ACCESS_LEVEL_STAFF,
        "content": (
            "Staff Clinical Standard Operating Procedure: Emergency Triage Protocols\n"
            "Classification: STAFF_ONLY | Effective Date: 2026-02-15 | Version: 3.0\n\n"
            "1. Triage Categories:\n"
            "   - Category 1 (Resuscitation): Immediate attention within 0 minutes. Cardiac arrest, airway obstruction.\n"
            "   - Category 2 (Emergency): Attention within 10 minutes. Severe chest pain, stroke symptoms, acute respiratory distress.\n"
            "   - Category 3 (Urgent): Attention within 30 minutes. Moderate trauma, acute abdominal pain.\n"
            "   - Category 4 (Semi-urgent): Attention within 60 minutes. Minor fractures, controlled bleeding.\n"
            "   - Category 5 (Non-urgent): Attention within 120 minutes. Chronic symptoms, mild rashes.\n"
            "2. Internal Emergency Extension: Dial 5555 for Code Blue activation.\n"
            "3. Medication Dispensation: Narcotic log books must be dual-signed by Registered Nurses."
        ),
        "uploaded_by": "chief_medical_officer",
    },
    {
        "filename": "hospital_executive_admin_protocol.txt",
        "access_level": SecurityConfig.ACCESS_LEVEL_RESTRICTED,
        "content": (
            "Hospital Executive Administration and Crisis Governance Protocol\n"
            "Classification: RESTRICTED_ADMIN | Effective Date: 2026-03-01 | Version: 1.4\n\n"
            "1. Incident Command Structure: In severe disasters, the Medical Superintendent assumes Command Post authority.\n"
            "2. Budgetary Contingency: Emergency equipment procurement up to $50,000 can be authorized directly by the Chief of Staff.\n"
            "3. IT Security Incident Protocol: In case of suspected electronic data compromise, the IT Security Lead must isolate "
            "the electronic health record subnet and notify the Ministry within 2 hours.\n"
            "4. Staff Credential Verification: Annual re-credentialing audits are conducted every November."
        ),
        "uploaded_by": "hospital_superintendent",
    },
]


def seed_database():
    """Ingests and approves sample hospital documents, then indexes into the vector store."""
    print("Seeding initial departmental documents...")
    for item in SAMPLE_DOCS:
        raw_bytes = item["content"].encode("utf-8")
        meta = document_ingestion_service.stage_upload(
            filename=item["filename"],
            file_bytes=raw_bytes,
            access_level=item["access_level"],
            uploaded_by=item["uploaded_by"],
            version="1.0"
        )
        approved_meta = document_ingestion_service.approve_document(
            doc_id=meta.doc_id,
            admin_user="system_admin",
            admin_role="ADMIN"
        )
        text = document_ingestion_service.get_document_text(approved_meta.doc_id)
        if text:
            vector_store_service.index_document(
                doc_id=approved_meta.doc_id,
                text=text,
                metadata={
                    "filename": approved_meta.filename,
                    "access_level": approved_meta.access_level,
                    "version": approved_meta.version,
                    "effective_date": approved_meta.effective_date,
                },
                admin_role="ADMIN"
            )
            print(f"  [+] Ingested, Approved, & Indexed: {approved_meta.filename} ({approved_meta.access_level})")


if __name__ == "__main__":
    seed_database()
