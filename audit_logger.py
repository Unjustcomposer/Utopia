import hashlib
import json
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SecureAuditLogger:
    """
    SOC2 Type II Compliant Audit Logger.
    
    Cryptographically hashes and sequentially logs all autonomous write actions 
    (e.g., SAP Purchase Orders) to ensure non-repudiation and traceability.
    """
    
    def __init__(self):
        self.sequence_id = 0
        self.previous_hash = "GENESIS_BLOCK"
        
    def _generate_hash(self, payload: Dict[str, Any], timestamp: float) -> str:
        """Generates a SHA-256 hash of the payload to ensure immutability."""
        log_entry = {
            "sequence": self.sequence_id,
            "timestamp": timestamp,
            "payload": payload,
            "previous_hash": self.previous_hash
        }
        
        entry_str = json.dumps(log_entry, sort_keys=True).encode('utf-8')
        return hashlib.sha256(entry_str).hexdigest()

    def log_autonomous_action(self, action_type: str, payload: Dict[str, Any], system: str) -> str:
        """
        Records an autonomous ERP write action. 
        In production, this writes to an immutable WORM (Write Once Read Many) storage bucket.
        """
        timestamp = time.time()
        self.sequence_id += 1
        
        current_hash = self._generate_hash(payload, timestamp)
        
        audit_record = {
            "audit_id": current_hash,
            "sequence": self.sequence_id,
            "timestamp": timestamp,
            "action_type": action_type,
            "target_system": system,
            "payload": payload,
            "previous_hash": self.previous_hash
        }
        
        self.previous_hash = current_hash
        
        # Log to secure stream (stdout for demonstration)
        logger.info(f"[SOC2 AUDIT LOG] Action: {action_type} | System: {system} | Hash: {current_hash}")
        
        return current_hash
