"""
Tests for VC Issuer
"""

import unittest
import json
from datetime import datetime, timedelta

from ..vc_issuer import (
    VCIssuer,
    VerifiableCredential,
    VerifiablePresentation,
    vc_issuer,
)
from ..did_manager import DIDManager


class TestVerifiableCredential(unittest.TestCase):
    """Test cases for Verifiable Credential"""

    def setUp(self):
        """Set up test fixtures"""
        self.did_manager = DIDManager()
        self.issuer_did = self.did_manager.create_did("terra")
        self.subject_did = self.did_manager.create_did("terra")

    def test_vc_creation(self):
        """Test VC creation"""
        claims = {"name": "Test Agent", "capability": "certification"}
        vc = VerifiableCredential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        self.assertEqual(vc.issuer, self.issuer_did)
        self.assertEqual(vc.subject, self.subject_did)
        self.assertIn("TestCredential", vc.type)
        self.assertEqual(vc.claims, claims)

    def test_vc_to_dict(self):
        """Test converting VC to dictionary"""
        claims = {"name": "Test Agent"}
        vc = VerifiableCredential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        vc_dict = vc.to_dict()

        self.assertIn("@context", vc_dict)
        self.assertIn("type", vc_dict)
        self.assertIn("issuer", vc_dict)
        self.assertIn("issuanceDate", vc_dict)
        self.assertIn("expirationDate", vc_dict)
        self.assertIn("credentialSubject", vc_dict)

    def test_vc_to_json(self):
        """Test converting VC to JSON"""
        claims = {"name": "Test Agent"}
        vc = VerifiableCredential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        vc_json = vc.to_json()
        self.assertIsInstance(vc_json, str)

        # Should be valid JSON
        parsed = json.loads(vc_json)
        self.assertEqual(parsed["issuer"], self.issuer_did)


class TestVCIssuer(unittest.TestCase):
    """Test cases for VC Issuer"""

    def setUp(self):
        """Set up test fixtures"""
        self.did_manager = DIDManager()
        self.vc_issuer = VCIssuer(self.did_manager)

        self.issuer_did = self.did_manager.create_did("terra")
        self.subject_did = self.did_manager.create_did("terra")

    def test_issue_credential(self):
        """Test issuing a credential"""
        claims = {"name": "Test Agent", "capability": "certification"}

        vc_jwt = self.vc_issuer.issue_credential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["CertificationCredential"],
            claims=claims,
        )

        self.assertIsNotNone(vc_jwt)
        self.assertIsInstance(vc_jwt, str)

        # Should have 3 parts (header.payload.signature)
        parts = vc_jwt.split(".")
        self.assertEqual(len(parts), 3)

    def test_verify_valid_credential(self):
        """Test verifying a valid credential"""
        claims = {"name": "Test Agent"}

        # Issue credential
        vc_jwt = self.vc_issuer.issue_credential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        # Verify credential
        is_valid = self.vc_issuer.verify_credential(vc_jwt)
        self.assertTrue(is_valid)

    def test_verify_invalid_credential(self):
        """Test verifying an invalid credential"""
        invalid_jwt = "invalid.jwt.signature"
        is_valid = self.vc_issuer.verify_credential(invalid_jwt)
        self.assertFalse(is_valid)

    def test_revoke_credential(self):
        """Test revoking a credential"""
        claims = {"name": "Test Agent"}

        # Issue credential
        vc_jwt = self.vc_issuer.issue_credential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        # Extract credential ID from JWT
        import base64

        payload_b64 = vc_jwt.split(".")[1]
        payload = json.loads(base64.b64decode(payload_b64 + "==").decode())
        credential_id = payload.get("jti")

        # Verify before revocation
        is_valid_before = self.vc_issuer.verify_credential(vc_jwt)
        self.assertTrue(is_valid_before)

        # Revoke credential
        self.vc_issuer.revoke_credential(credential_id)

        # Verify after revocation
        is_valid_after = self.vc_issuer.verify_credential(vc_jwt)
        self.assertFalse(is_valid_after)

    def test_create_presentation(self):
        """Test creating a verifiable presentation"""
        claims = {"name": "Test Agent"}

        # Issue credential
        vc_jwt = self.vc_issuer.issue_credential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        # Create presentation
        presentation = self.vc_issuer.create_presentation(
            holder_did=self.subject_did, credentials=[vc_jwt]
        )

        self.assertIsNotNone(presentation)
        self.assertIsInstance(presentation, VerifiablePresentation)
        self.assertEqual(presentation.holder, self.subject_did)
        self.assertEqual(len(presentation.credentials), 1)

    def test_presentation_to_dict(self):
        """Test converting presentation to dictionary"""
        claims = {"name": "Test Agent"}

        # Issue credential
        vc_jwt = self.vc_issuer.issue_credential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        # Create presentation
        presentation = self.vc_issuer.create_presentation(
            holder_did=self.subject_did, credentials=[vc_jwt]
        )

        pres_dict = presentation.to_dict()

        self.assertIn("@context", pres_dict)
        self.assertIn("type", pres_dict)
        self.assertIn("holder", pres_dict)
        self.assertIn("verifiableCredential", pres_dict)
        self.assertEqual(len(pres_dict["verifiableCredential"]), 1)

    def test_get_issued_credentials(self):
        """Test getting issued credentials"""
        claims = {"name": "Test Agent"}

        # Issue credential
        self.vc_issuer.issue_credential(
            issuer_did=self.issuer_did,
            subject_did=self.subject_did,
            credential_type=["TestCredential"],
            claims=claims,
        )

        # Get all issued credentials
        all_credentials = self.vc_issuer.get_issued_credentials()
        self.assertEqual(len(all_credentials), 1)

        # Get credentials for specific subject
        subject_credentials = self.vc_issuer.get_issued_credentials(self.subject_did)
        self.assertEqual(len(subject_credentials), 1)

        # Get credentials for non-existent subject
        no_credentials = self.vc_issuer.get_issued_credentials("did:terra:nonexistent")
        self.assertEqual(len(no_credentials), 0)


if __name__ == "__main__":
    unittest.main()
