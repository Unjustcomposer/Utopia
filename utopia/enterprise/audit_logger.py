import hashlib
import json
import logging
import time
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AuditLogger:
    """
    Standard Audit Logger.
    
    Logs autonomous write actions (e.g., SAP Purchase Orders) for traceability.
    """
    
    def __init__(self):
        self.sequence_id = 0
        self.previous_hash = "0" * 64
        self._load_last_state()
        
    def _load_last_state(self):
        log_path = "data/audit_log.jsonl"
        if not os.path.exists(log_path):
            return
            
        last_line = None
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        last_line = line
        except Exception as e:
            logger.error(f"Failed to read durable audit log: {e}")
            return
            
        if last_line:
            try:
                entry = json.loads(last_line)
                self.sequence_id = entry.get("sequence_id", 0)
                self.previous_hash = entry.get("hash", "0" * 64)
            except json.JSONDecodeError:
                pass
        
    def log_autonomous_action(self, action_type: str, payload: Dict[str, Any], system: str) -> None:
        """
        Records an autonomous ERP write action. 
        """
        timestamp = time.time()
        self.sequence_id += 1
        
        payload_str = json.dumps(payload, sort_keys=True)
        raw_string = f"{self.previous_hash}|{timestamp}|{self.sequence_id}|{action_type}|{system}|{payload_str}"
        current_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        
        # Log to standard output stream and durable append-only file
        log_entry = {
            "timestamp": timestamp,
            "sequence_id": self.sequence_id,
            "action": action_type,
            "system": system,
            "payload": payload,
            "previous_hash": self.previous_hash,
            "hash": current_hash
        }
        
        self.previous_hash = current_hash
        logger.info(f"[AUDIT LOG] Action: {action_type} | System: {system} | Sequence: {self.sequence_id}")
        
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/audit_log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to durable audit log: {e}")

# Global singleton instance
audit_logger = AuditLogger()
