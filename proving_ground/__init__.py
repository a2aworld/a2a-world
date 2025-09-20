"""
Proving Ground Module for Verifiable Credentials and Certification

This module implements a certification framework using W3C Verifiable Credentials
and Decentralized Identifiers for the Terra Constellata project. It provides
secure agent certification, VC issuance/verification, and DID management.

Components:
- ProvingGround_Manager: Main agent for administering certifications
- VC Issuer: Handles issuance of W3C Verifiable Credentials
- DID Manager: Manages Decentralized Identifiers
- Certification Tests: Framework for testing agent capabilities
"""

__version__ = "0.1.0"

from .manager import ProvingGroundManager
from .vc_issuer import VCIssuer
from .did_manager import DIDManager
from .certification_tests import CertificationTestSuite

__all__ = ["ProvingGroundManager", "VCIssuer", "DIDManager", "CertificationTestSuite"]
