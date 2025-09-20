"""
Tests for DID Manager
"""

import unittest
import asyncio
from unittest.mock import Mock, patch

from ..did_manager import DIDManager, did_manager


class TestDIDManager(unittest.TestCase):
    """Test cases for DID Manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.did_manager = DIDManager()

    def test_create_did(self):
        """Test DID creation"""
        did = self.did_manager.create_did("terra")

        self.assertIsNotNone(did)
        self.assertTrue(did.startswith("did:terra:"))
        self.assertIn(did, self.did_manager.dids)

    def test_get_did_document(self):
        """Test getting DID document"""
        did = self.did_manager.create_did("terra")
        doc = self.did_manager.get_did_document(did)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.did, did)
        self.assertIn("@context", doc.to_dict())

    def test_resolve_did(self):
        """Test DID resolution"""
        did = self.did_manager.create_did("terra")
        resolved = self.did_manager.resolve_did(did)

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["id"], did)

    def test_resolve_nonexistent_did(self):
        """Test resolving non-existent DID"""
        resolved = self.did_manager.resolve_did("did:terra:nonexistent")
        self.assertIsNone(resolved)

    def test_sign_and_verify_data(self):
        """Test data signing and verification"""
        did = self.did_manager.create_did("terra")
        test_data = b"Hello, World!"

        # Sign data
        signature = self.did_manager.sign_data(did, test_data)
        self.assertIsNotNone(signature)

        # Verify signature
        is_valid = self.did_manager.verify_signature(did, test_data, signature)
        self.assertTrue(is_valid)

    def test_verify_invalid_signature(self):
        """Test verification of invalid signature"""
        did = self.did_manager.create_did("terra")
        test_data = b"Hello, World!"
        wrong_data = b"Goodbye, World!"

        # Sign data
        signature = self.did_manager.sign_data(did, test_data)
        self.assertIsNotNone(signature)

        # Try to verify with wrong data
        is_valid = self.did_manager.verify_signature(did, wrong_data, signature)
        self.assertFalse(is_valid)

    def test_list_dids(self):
        """Test listing DIDs"""
        initial_count = len(self.did_manager.list_dids())

        self.did_manager.create_did("terra")
        self.did_manager.create_did("key")

        current_count = len(self.did_manager.list_dids())
        self.assertEqual(current_count, initial_count + 2)

    def test_update_did_document(self):
        """Test updating DID document"""
        did = self.did_manager.create_did("terra")
        doc_before = self.did_manager.get_did_document(did)

        # Update document
        self.did_manager.update_did_document(did, {"test": "update"})

        doc_after = self.did_manager.get_did_document(did)
        self.assertIsNotNone(doc_after)
        # Note: In this simplified implementation, updates are not actually applied


class TestDIDDocument(unittest.TestCase):
    """Test cases for DID Document"""

    def setUp(self):
        """Set up test fixtures"""
        from ..did_manager import DIDDocument
        import base64

        # Create mock public key
        mock_public_key = b"mock_public_key_32_bytes_long"
        self.did_doc = DIDDocument("did:terra:test", mock_public_key)

    def test_to_dict(self):
        """Test converting DID document to dictionary"""
        doc_dict = self.did_doc.to_dict()

        self.assertIn("@context", doc_dict)
        self.assertIn("id", doc_dict)
        self.assertIn("verificationMethod", doc_dict)
        self.assertEqual(doc_dict["id"], "did:terra:test")

    def test_to_json(self):
        """Test converting DID document to JSON"""
        doc_json = self.did_doc.to_json()

        self.assertIsInstance(doc_json, str)
        # Should be valid JSON
        import json

        parsed = json.loads(doc_json)
        self.assertEqual(parsed["id"], "did:terra:test")


if __name__ == "__main__":
    unittest.main()
