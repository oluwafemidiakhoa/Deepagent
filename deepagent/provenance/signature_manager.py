"""
Signature Manager - Foundation #3

Cryptographic signing and verification of data.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class DataSignature:
    """Cryptographic signature for data"""
    data_id: str
    signature: str
    algorithm: str
    signed_at: datetime
    signer_id: str


@dataclass
class VerificationResult:
    """Result of signature verification"""
    verified: bool
    data_id: str
    signature_valid: bool
    data_tampered: bool
    message: str


class SignatureManager:
    """Manages cryptographic signatures"""

    def __init__(self, signer_id: str = "deepagent"):
        self.signer_id = signer_id
        self.signatures: Dict[str, DataSignature] = {}

    def sign_data(self, data_id: str, data: Any) -> DataSignature:
        """Create cryptographic signature"""
        data_str = json.dumps(data, sort_keys=True)
        signature = hashlib.sha256(data_str.encode()).hexdigest()

        sig = DataSignature(
            data_id=data_id,
            signature=signature,
            algorithm="sha256",
            signed_at=datetime.now(),
            signer_id=self.signer_id
        )

        self.signatures[data_id] = sig
        return sig

    def verify_data(self, data_id: str, data: Any) -> VerificationResult:
        """Verify data signature"""
        if data_id not in self.signatures:
            return VerificationResult(
                verified=False,
                data_id=data_id,
                signature_valid=False,
                data_tampered=False,
                message="No signature found"
            )

        sig = self.signatures[data_id]
        data_str = json.dumps(data, sort_keys=True)
        current_sig = hashlib.sha256(data_str.encode()).hexdigest()

        matches = current_sig == sig.signature

        return VerificationResult(
            verified=matches,
            data_id=data_id,
            signature_valid=True,
            data_tampered=not matches,
            message="Valid" if matches else "Data has been tampered"
        )


from typing import Dict
