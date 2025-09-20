"""
DID Manager for Decentralized Identifiers

This module provides functionality for creating, managing, and resolving
Decentralized Identifiers (DIDs) following W3C DID specifications.
"""

import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64
import hashlib

logger = logging.getLogger(__name__)


class DIDDocument:
    """W3C DID Document representation"""

    def __init__(self, did: str, public_key: bytes, controller: str = None):
        self.did = did
        self.public_key = public_key
        self.controller = controller or did
        self.created = datetime.now(timezone.utc).isoformat()
        self.updated = self.created

    def to_dict(self) -> Dict[str, Any]:
        """Convert DID Document to dictionary"""
        # Convert public key to multibase format (simplified)
        public_key_multibase = base64.b64encode(self.public_key).decode("utf-8")

        return {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/suites/ed25519-2020/v1",
            ],
            "id": self.did,
            "controller": self.controller,
            "verificationMethod": [
                {
                    "id": f"{self.did}#key-1",
                    "type": "Ed25519VerificationKey2020",
                    "controller": self.controller,
                    "publicKeyMultibase": f"z{public_key_multibase}",
                }
            ],
            "authentication": [f"{self.did}#key-1"],
            "assertionMethod": [f"{self.did}#key-1"],
            "created": self.created,
            "updated": self.updated,
        }

    def to_json(self) -> str:
        """Convert DID Document to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class DIDManager:
    """Manager for Decentralized Identifiers"""

    def __init__(self):
        self.dids: Dict[str, DIDDocument] = {}
        self.private_keys: Dict[str, ed25519.Ed25519PrivateKey] = {}
        self.resolver_cache: Dict[str, DIDDocument] = {}

    def create_did(self, method: str = "terra") -> str:
        """
        Create a new DID with the specified method.

        Args:
            method: DID method (e.g., 'terra', 'key', 'web')

        Returns:
            The created DID string
        """
        # Generate Ed25519 key pair
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Serialize public key
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        # Create DID identifier from public key hash
        key_hash = hashlib.sha256(public_bytes).digest()
        did_identifier = (
            base64.urlsafe_b64encode(key_hash[:16]).decode("utf-8").rstrip("=")
        )

        # Create DID
        did = f"did:{method}:{did_identifier}"

        # Create DID Document
        did_doc = DIDDocument(did, public_bytes)

        # Store DID and private key
        self.dids[did] = did_doc
        self.private_keys[did] = private_key

        logger.info(f"Created DID: {did}")
        return did

    def get_did_document(self, did: str) -> Optional[DIDDocument]:
        """
        Get the DID Document for a given DID.

        Args:
            did: The DID to resolve

        Returns:
            DIDDocument if found, None otherwise
        """
        return self.dids.get(did)

    def resolve_did(self, did: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a DID to its DID Document.

        Args:
            did: The DID to resolve

        Returns:
            DID Document as dictionary if found, None otherwise
        """
        did_doc = self.get_did_document(did)
        if did_doc:
            return did_doc.to_dict()
        return None

    def get_private_key(self, did: str) -> Optional[ed25519.Ed25519PrivateKey]:
        """
        Get the private key for a DID.

        Args:
            did: The DID

        Returns:
            Private key if found, None otherwise
        """
        return self.private_keys.get(did)

    def sign_data(self, did: str, data: bytes) -> Optional[str]:
        """
        Sign data using the private key associated with a DID.

        Args:
            did: The DID to use for signing
            data: Data to sign

        Returns:
            Base64-encoded signature if successful, None otherwise
        """
        private_key = self.get_private_key(did)
        if not private_key:
            logger.error(f"No private key found for DID: {did}")
            return None

        signature = private_key.sign(data)
        return base64.b64encode(signature).decode("utf-8")

    def verify_signature(self, did: str, data: bytes, signature: str) -> bool:
        """
        Verify a signature using the public key associated with a DID.

        Args:
            did: The DID
            data: Original data
            signature: Base64-encoded signature

        Returns:
            True if signature is valid, False otherwise
        """
        did_doc = self.get_did_document(did)
        if not did_doc:
            logger.error(f"No DID document found for: {did}")
            return False

        try:
            signature_bytes = base64.b64decode(signature)
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(did_doc.public_key)
            public_key.verify(signature_bytes, data)
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def list_dids(self) -> list[str]:
        """List all managed DIDs"""
        return list(self.dids.keys())

    def update_did_document(self, did: str, updates: Dict[str, Any]):
        """
        Update a DID Document.

        Args:
            did: The DID to update
            updates: Dictionary of updates to apply
        """
        did_doc = self.get_did_document(did)
        if did_doc:
            # Apply updates (simplified - in practice would be more complex)
            did_doc.updated = datetime.now(timezone.utc).isoformat()
            logger.info(f"Updated DID document for: {did}")
        else:
            logger.error(f"DID not found: {did}")


# Global DID manager instance
did_manager = DIDManager()
