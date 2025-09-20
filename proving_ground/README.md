# Proving Ground Module

The Proving Ground module implements a comprehensive certification framework for the Terra Constellata project using W3C Verifiable Credentials and Decentralized Identifiers.

## Overview

This module provides:
- **DID Management**: Creation, resolution, and management of Decentralized Identifiers
- **VC Issuance**: Issuance and verification of W3C Verifiable Credentials
- **Agent Certification**: Automated testing and certification of agents
- **A2A Integration**: Secure communication via the Agent-to-Agent protocol

## Components

### ProvingGround_Manager
The main agent that orchestrates the certification process. It:
- Administers certification tests
- Issues verifiable credentials
- Manages DIDs
- Integrates with the A2A protocol

### DID Manager
Handles Decentralized Identifiers:
- Creates DIDs with Ed25519 key pairs
- Resolves DID documents
- Signs and verifies data
- Manages DID document lifecycle

### VC Issuer
Manages Verifiable Credentials:
- Issues W3C-compliant VCs
- Verifies credential signatures
- Supports credential revocation
- Creates verifiable presentations

### Certification Tests
Comprehensive test suite for agent capabilities:
- A2A protocol communication tests
- Task processing tests
- Security compliance tests
- Customizable test frameworks

## Usage

### Basic Usage

```python
from proving_ground import ProvingGroundManager, did_manager, vc_issuer

# Create the manager
manager = ProvingGroundManager(
    name="ProvingGround",
    llm=your_llm,
    tools=your_tools
)

# Create a DID
did = did_manager.create_did("terra")
print(f"Created DID: {did}")

# Issue a credential
vc = vc_issuer.issue_credential(
    issuer_did=did,
    subject_did="did:terra:agent:subject",
    credential_type=["CertificationCredential"],
    claims={"name": "Certified Agent", "level": "Advanced"}
)

# Process certification
result = await manager.process_task(
    "Certify agent test_agent",
    agent_id="test_agent_id",
    agent_name="Test Agent"
)
```

### A2A Integration

The ProvingGround Manager integrates with the A2A protocol for secure agent communication:

```python
# Connect to A2A server
await manager.connect_a2a()

# Send certification request
result = await manager.send_message(
    CertificationRequest(
        sender_agent="certifier",
        subject="agent_to_certify",
        certification_type="agent_certification",
        evidence={"performance": "excellent"},
        criteria=["communication", "security"]
    ),
    target_agent="target_agent"
)
```

## Architecture

```
proving_ground/
├── __init__.py          # Module exports
├── manager.py           # ProvingGround_Manager agent
├── did_manager.py       # DID management
├── vc_issuer.py         # VC issuance and verification
├── certification_tests.py # Test suite framework
├── tests/               # Test files
│   ├── __init__.py
│   ├── test_did_manager.py
│   ├── test_vc_issuer.py
│   ├── test_manager.py
│   └── test_a2a_integration.py
└── README.md           # This file
```

## Security Features

- **Cryptographic Signing**: Uses Ed25519 for secure signing
- **Credential Revocation**: Support for revoking compromised credentials
- **Expiration Management**: Automatic credential expiration
- **Audit Logging**: Comprehensive logging of all operations

## Testing

Run the test suite:

```bash
python -m pytest proving_ground/tests/
```

Or run individual test files:

```bash
python proving_ground/tests/test_did_manager.py
python proving_ground/tests/test_vc_issuer.py
python proving_ground/tests/test_manager.py
python proving_ground/tests/test_a2a_integration.py
```

## Dependencies

- cryptography: For cryptographic operations
- pydantic: For data validation
- langchain: For agent functionality
- asyncio: For asynchronous operations

## Standards Compliance

- **W3C DID Core**: Decentralized Identifier specification
- **W3C VC Data Model**: Verifiable Credentials data model
- **Ed25519 Signatures**: Secure cryptographic signatures
- **JWT Format**: JSON Web Token format for credentials

## Future Enhancements

- Support for additional DID methods (key, web, etc.)
- Integration with external VC registries
- Advanced revocation mechanisms
- Multi-signature credentials
- Zero-knowledge proofs