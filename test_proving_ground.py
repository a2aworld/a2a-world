#!/usr/bin/env python3
"""
Simple test script for Proving Ground functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from proving_ground.did_manager import DIDManager, did_manager
from proving_ground.vc_issuer import VCIssuer, vc_issuer
from proving_ground.certification_tests import (
    CertificationTestSuite,
    create_default_test_suite,
)


def test_did_functionality():
    """Test DID creation and management"""
    print("Testing DID functionality...")

    # Create a DID
    did = did_manager.create_did("terra")
    print(f"Created DID: {did}")

    # Get DID document
    doc = did_manager.get_did_document(did)
    print(f"DID document created: {doc is not None}")

    # Test signing and verification
    test_data = b"Hello, World!"
    signature = did_manager.sign_data(did, test_data)
    print(f"Signature created: {signature is not None}")

    is_valid = did_manager.verify_signature(did, test_data, signature)
    print(f"Signature verified: {is_valid}")

    print("DID functionality test completed.\n")


def test_vc_functionality():
    """Test VC issuance and verification"""
    print("Testing VC functionality...")

    # Create DIDs
    issuer_did = did_manager.create_did("terra")
    subject_did = did_manager.create_did("terra")

    # Issue credential
    claims = {"name": "Test Agent", "capability": "certification"}
    vc_jwt = vc_issuer.issue_credential(
        issuer_did=issuer_did,
        subject_did=subject_did,
        credential_type=["TestCredential"],
        claims=claims,
    )
    print(f"VC issued: {vc_jwt is not None}")

    # Verify credential
    is_valid = vc_issuer.verify_credential(vc_jwt)
    print(f"VC verified: {is_valid}")

    print("VC functionality test completed.\n")


def test_certification_tests():
    """Test certification test suite"""
    print("Testing certification test suite...")

    # Create test suite
    suite = create_default_test_suite()
    print(f"Test suite created with {len(suite.tests)} tests")

    # Mock agent context
    agent_context = {
        "agent_id": "test_agent",
        "agent_name": "Test Agent",
        "a2a_client": None,
        "agent": None,
        "security_features": ["encryption", "authentication"],
    }

    print("Certification test suite created successfully.\n")


def main():
    """Run all tests"""
    print("Starting Proving Ground functionality tests...\n")

    try:
        test_did_functionality()
        test_vc_functionality()
        test_certification_tests()

        print("All tests completed successfully!")
        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
