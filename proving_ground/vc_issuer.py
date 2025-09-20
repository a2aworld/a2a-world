"""
VC Issuer for W3C Verifiable Credentials

This module provides functionality for issuing and verifying W3C Verifiable Credentials
following the W3C VC Data Model specification.
"""

import json
import logging
import base64
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from uuid import uuid4

from .did_manager import DIDManager, did_manager

logger = logging.getLogger(__name__)


class VerifiableCredential:
    """W3C Verifiable Credential representation"""

    def __init__(
        self,
        issuer_did: str,
        subject_did: str,
        credential_type: List[str],
        claims: Dict[str, Any],
        expiration_date: Optional[datetime] = None,
        issuance_date: Optional[datetime] = None,
    ):
        self.id = f"urn:uuid:{uuid4()}"
        self.issuer = issuer_did
        self.subject = subject_did
        self.type = ["VerifiableCredential"] + credential_type
        self.claims = claims
        self.issuance_date = issuance_date or datetime.utcnow()
        self.expiration_date = expiration_date or (
            self.issuance_date + timedelta(days=365)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert VC to dictionary"""
        return {
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://www.w3.org/2018/credentials/examples/v1",
            ],
            "id": self.id,
            "type": self.type,
            "issuer": self.issuer,
            "issuanceDate": self.issuance_date.isoformat() + "Z",
            "expirationDate": self.expiration_date.isoformat() + "Z",
            "credentialSubject": {"id": self.subject, **self.claims},
        }

    def to_json(self) -> str:
        """Convert VC to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class VerifiablePresentation:
    """W3C Verifiable Presentation representation"""

    def __init__(self, holder_did: str, credentials: List[VerifiableCredential]):
        self.id = f"urn:uuid:{uuid4()}"
        self.holder = holder_did
        self.credentials = credentials
        self.created = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert VP to dictionary"""
        return {
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "id": self.id,
            "type": ["VerifiablePresentation"],
            "holder": self.holder,
            "verifiableCredential": [vc.to_dict() for vc in self.credentials],
            "created": self.created.isoformat() + "Z",
        }


class VCIssuer:
    """Issuer for W3C Verifiable Credentials"""

    def __init__(self, did_manager: DIDManager = None):
        self.did_manager = did_manager or did_manager
        self.issued_credentials: Dict[str, VerifiableCredential] = {}
        self.revoked_credentials: set[str] = set()

    def issue_credential(
        self,
        issuer_did: str,
        subject_did: str,
        credential_type: List[str],
        claims: Dict[str, Any],
        validity_days: int = 365,
    ) -> Optional[str]:
        """
        Issue a new Verifiable Credential.

        Args:
            issuer_did: DID of the issuer
            subject_did: DID of the subject
            credential_type: Types of the credential
            claims: Claims to include in the credential
            validity_days: Number of days the credential is valid

        Returns:
            Signed JWT credential if successful, None otherwise
        """
        # Verify issuer DID exists
        if not self.did_manager.get_did_document(issuer_did):
            logger.error(f"Issuer DID not found: {issuer_did}")
            return None

        # Create credential
        expiration_date = datetime.utcnow() + timedelta(days=validity_days)
        vc = VerifiableCredential(
            issuer_did=issuer_did,
            subject_did=subject_did,
            credential_type=credential_type,
            claims=claims,
            expiration_date=expiration_date,
        )

        # Sign the credential
        signed_vc = self._sign_credential(vc)
        if signed_vc:
            self.issued_credentials[vc.id] = vc
            logger.info(f"Issued credential: {vc.id} for subject: {subject_did}")
            return signed_vc

        return None

    def _sign_credential(self, vc: VerifiableCredential) -> Optional[str]:
        """
        Sign a Verifiable Credential using the issuer's private key.

        Args:
            vc: The credential to sign

        Returns:
            Signed JWT string if successful, None otherwise
        """
        # Create JWT payload
        payload = vc.to_dict()
        payload["jti"] = vc.id  # JWT ID

        # Convert to JSON and sign
        vc_json = json.dumps(payload, separators=(",", ":"))
        signature = self.did_manager.sign_data(vc.issuer, vc_json.encode("utf-8"))

        if signature:
            # Create JWT (simplified - in practice would use proper JWT library)
            header = {"alg": "EdDSA", "typ": "JWT"}
            header_b64 = (
                base64.b64encode(json.dumps(header).encode()).decode().rstrip("=")
            )
            payload_b64 = base64.b64encode(vc_json.encode()).decode().rstrip("=")
            jwt = f"{header_b64}.{payload_b64}.{signature}"
            return jwt

        return None

    def verify_credential(self, signed_credential: str) -> bool:
        """
        Verify a signed Verifiable Credential.

        Args:
            signed_credential: The signed credential JWT

        Returns:
            True if valid, False otherwise
        """
        try:
            # Parse JWT (simplified)
            parts = signed_credential.split(".")
            if len(parts) != 3:
                return False

            header_b64, payload_b64, signature = parts
            vc_json = base64.b64decode(payload_b64 + "==").decode("utf-8")
            vc_data = json.loads(vc_json)

            # Check if revoked
            if vc_data.get("jti") in self.revoked_credentials:
                logger.warning(f"Credential revoked: {vc_data.get('jti')}")
                return False

            # Check expiration
            exp_date = datetime.fromisoformat(vc_data["expirationDate"][:-1])
            if datetime.utcnow() > exp_date:
                logger.warning(f"Credential expired: {vc_data.get('jti')}")
                return False

            # Verify signature
            issuer_did = vc_data["issuer"]
            return self.did_manager.verify_signature(
                issuer_did, f"{header_b64}.{payload_b64}".encode(), signature
            )

        except Exception as e:
            logger.error(f"Credential verification failed: {e}")
            return False

    def revoke_credential(self, credential_id: str):
        """
        Revoke a credential.

        Args:
            credential_id: ID of the credential to revoke
        """
        self.revoked_credentials.add(credential_id)
        logger.info(f"Revoked credential: {credential_id}")

    def create_presentation(
        self, holder_did: str, credentials: List[str]
    ) -> Optional[VerifiablePresentation]:
        """
        Create a Verifiable Presentation from credentials.

        Args:
            holder_did: DID of the holder
            credentials: List of signed credential JWTs

        Returns:
            VerifiablePresentation if successful, None otherwise
        """
        vc_objects = []
        for cred in credentials:
            if not self.verify_credential(cred):
                logger.error("Invalid credential in presentation")
                return None

            # Parse credential to get VC object
            parts = cred.split(".")
            payload_b64 = parts[1]
            vc_json = base64.b64decode(payload_b64 + "==").decode("utf-8")
            vc_data = json.loads(vc_json)

            # Reconstruct VC object (simplified)
            vc = VerifiableCredential(
                issuer_did=vc_data["issuer"],
                subject_did=vc_data["credentialSubject"]["id"],
                credential_type=vc_data["type"][1:],  # Remove "VerifiableCredential"
                claims={
                    k: v for k, v in vc_data["credentialSubject"].items() if k != "id"
                },
                issuance_date=datetime.fromisoformat(vc_data["issuanceDate"][:-1]),
                expiration_date=datetime.fromisoformat(vc_data["expirationDate"][:-1]),
            )
            vc.id = vc_data["id"]
            vc_objects.append(vc)

        presentation = VerifiablePresentation(holder_did, vc_objects)
        logger.info(f"Created presentation for holder: {holder_did}")
        return presentation

    def get_issued_credentials(
        self, subject_did: Optional[str] = None
    ) -> List[VerifiableCredential]:
        """
        Get issued credentials, optionally filtered by subject.

        Args:
            subject_did: Optional subject DID filter

        Returns:
            List of issued credentials
        """
        if subject_did:
            return [
                vc
                for vc in self.issued_credentials.values()
                if vc.subject == subject_did
            ]
        return list(self.issued_credentials.values())


# Global VC issuer instance
vc_issuer = VCIssuer()
