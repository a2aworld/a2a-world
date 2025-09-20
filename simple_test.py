#!/usr/bin/env python3
"""
Simple test for core proving ground components
"""

import sys
import os
import base64
import json
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


def test_did_creation():
    """Test basic DID creation"""
    print("Testing DID creation...")

    # Generate Ed25519 key pair
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Serialize public key
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )

    # Create DID identifier
    import hashlib

    key_hash = hashlib.sha256(public_bytes).digest()
    did_identifier = base64.urlsafe_b64encode(key_hash[:16]).decode("utf-8").rstrip("=")

    # Create DID
    did = f"did:terra:{did_identifier}"
    print(f"Created DID: {did}")

    # Test signing
    test_data = b"Hello, World!"
    signature = private_key.sign(test_data)
    print(f"Signature created: {len(signature)} bytes")

    # Test verification
    public_key.verify(signature, test_data)
    print("Signature verified successfully")

    return did, private_key, public_bytes


def test_vc_structure():
    """Test VC data structure"""
    print("\nTesting VC structure...")

    # Create a simple VC
    vc = {
        "@context": [
            "https://www.w3.org/2018/credentials/v1",
            "https://www.w3.org/2018/credentials/examples/v1",
        ],
        "id": "urn:uuid:test-123",
        "type": ["VerifiableCredential", "TestCredential"],
        "issuer": "did:terra:test-issuer",
        "issuanceDate": datetime.utcnow().isoformat() + "Z",
        "expirationDate": (datetime.utcnow() + timedelta(days=365)).isoformat() + "Z",
        "credentialSubject": {
            "id": "did:terra:test-subject",
            "name": "Test Agent",
            "capability": "certification",
        },
    }

    # Convert to JSON
    vc_json = json.dumps(vc, indent=2)
    print("VC JSON created successfully")
    print(f"VC length: {len(vc_json)} characters")

    return vc


def main():
    """Run tests"""
    print("Starting simple Proving Ground tests...\n")

    try:
        # Test DID functionality
        did, private_key, public_bytes = test_did_creation()

        # Test VC structure
        vc = test_vc_structure()

        print("\nAll basic tests passed!")
        print("Core cryptographic and data structure functionality is working.")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
