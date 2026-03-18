import os
import base64
import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from functools import wraps
import uuid

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import kms
    KMIP_AVAILABLE = True
except ImportException:
    KMIP_AVAILABLE = False


logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    pass


class KeyManagementError(Exception):
    pass


class AccessControlError(Exception):
    pass


class ComplianceError(Exception):
    pass


class AuditLogError(Exception):
    pass


class Role(str, Enum):
    ADMIN = "admin"
    WRITER = "writer"
    READER = "reader"
    AUDITOR = "auditor"
    NONE = "none"


class ComplianceRegulation(str, Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


class AuditEventType(str, Enum):
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    KEY_GENERATE = "key_generate"
    KEY_ROTATE = "key_rotate"
    KEY_DELETE = "key_delete"
    ACCESS_GRANT = "access_grant"
    ACCESS_REVOKE = "access_revoke"
    MEMORY_CREATE = "memory_create"
    MEMORY_READ = "memory_read"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_SEARCH = "memory_search"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE_DELETE = "compliance_delete"


@dataclass
class EncryptionKey:
    key_id: str
    agent_id: str
    key_material: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    algorithm: str = "AES-256"
    status: str = "active"


@dataclass
class AccessPolicy:
    role: Role
    agent_id: str
    namespace: str
    permissions: List[str]
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class AuditEntry:
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    agent_id: str
    namespace: str
    resource_id: Optional[str]
    actor_id: str
    action: str
    success: bool
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class MemoryExpirationPolicy:
    policy_id: str
    namespace: str
    retention_days: int
    encryption_key_id: str
    created_at: datetime
    last_enforced: Optional[datetime] = None
    regulation: Optional[ComplianceRegulation] = None
    auto_delete: bool = True
    notify_before_deletion: bool = True
    notification_days: int = 7


class KeyDerivationFunction:
    def __init__(self, salt: bytes, iterations: int = 100000):
        self.salt = salt
        self.iterations = iterations

    def derive(self, password: str, key_length: int = 32) -> bytes:
        if not CRYPTO_AVAILABLE:
            raise EncryptionError("cryptography library not available")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=self.salt,
            iterations=self.iterations,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))


class BaseKeyStore(ABC):
    @abstractmethod
    def store_key(self, encryption_key: EncryptionKey) -> bool:
        pass

    @abstractmethod
    def retrieve_key(self, key_id: str) -> Optional[EncryptionKey]:
        pass

    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        pass

    @abstractmethod
    def list_keys(self, agent_id: Optional[str] = None) -> List[EncryptionKey]:
        pass

    @abstractmethod
    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        pass


class SQLiteKeyStore(BaseKeyStore):
    def __init__(self, db_path: str = ":memory:", master_key: Optional[str] = None):
        self.db_path = db_path
        self.master_key = master_key or os.environ.get("MEM0_MASTER_KEY", secrets.token_hex(32))
        self._connection_pool = {}
        self._lock = threading.RLock()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        if thread_id not in self._connection_pool:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connection_pool[thread_id] = conn
        return self._connection_pool[thread_id]

    def _init_database(self):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    key_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    key_material_encrypted BLOB NOT NULL,
                    salt BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    rotated_at TEXT,
                    algorithm TEXT DEFAULT 'AES-256',
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_keys_agent_id ON encryption_keys(agent_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_keys_status ON encryption_keys(status)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS key_audit_log (
                    log_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    actor_id TEXT,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    details TEXT
                )
            """)

            conn.commit()

    def _encrypt_master(self, data: bytes, salt: bytes) -> bytes:
        kdf = KeyDerivationFunction(salt)
        key = kdf.derive(self.master_key)
        if not CRYPTO_AVAILABLE:
            return base64.b64encode(data)
        fernet = Fernet(key)
        return fernet.encrypt(data)

    def _decrypt_master(self, encrypted_data: bytes, salt: bytes) -> bytes:
        kdf = KeyDerivationFunction(salt)
        key = kdf.derive(self.master_key)
        if not CRYPTO_AVAILABLE:
            return base64.b64decode(encrypted_data)
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)

    def store_key(self, encryption_key: EncryptionKey) -> bool:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                salt = secrets.token_bytes(32)
                encrypted_material = self._encrypt_master(encryption_key.key_material, salt)

                metadata = json.dumps({
                    "algorithm": encryption_key.algorithm,
                    "created_by": "system"
                })

                cursor.execute("""
                    INSERT OR REPLACE INTO encryption_keys
                    (key_id, agent_id, key_material_encrypted, salt, created_at,
                     expires_at, rotated_at, algorithm, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    encryption_key.key_id,
                    encryption_key.agent_id,
                    encrypted_material,
                    salt,
                    encryption_key.created_at.isoformat(),
                    encryption_key.expires_at.isoformat() if encryption_key.expires_at else None,
                    encryption_key.rotated_at.isoformat() if encryption_key.rotated_at else None,
                    encryption_key.algorithm,
                    encryption_key.status,
                    metadata
                ))

                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to store key: {e}")
                return False

    def retrieve_key(self, key_id: str) -> Optional[EncryptionKey]:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM encryption_keys WHERE key_id = ? AND status = 'active'
                """, (key_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                salt = bytes(row["salt"])
                encrypted_material = bytes(row["key_material_encrypted"])
                key_material = self._decrypt_master(encrypted_material, salt)

                return EncryptionKey(
                    key_id=row["key_id"],
                    agent_id=row["agent_id"],
                    key_material=key_material,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                    rotated_at=datetime.fromisoformat(row["rotated_at"]) if row["rotated_at"] else None,
                    algorithm=row["algorithm"],
                    status=row["status"]
                )
            except Exception as e:
                logger.error(f"Failed to retrieve key: {e}")
                return None

    def delete_key(self, key_id: str) -> bool:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE encryption_keys SET status = 'deleted' WHERE key_id = ?
                """, (key_id,))

                cursor.execute("""
                    INSERT INTO key_audit_log (log_id, key_id, operation, timestamp, success)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    key_id,
                    "delete",
                    datetime.utcnow().isoformat(),
                    1
                ))

                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to delete key: {e}")
                return False

    def list_keys(self, agent_id: Optional[str] = None) -> List[EncryptionKey]:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                if agent_id:
                    cursor.execute("""
                        SELECT * FROM encryption_keys
                        WHERE agent_id = ? AND status = 'active'
                    """, (agent_id,))
                else:
                    cursor.execute("""
                        SELECT * FROM encryption_keys WHERE status = 'active'
                    """)

                keys = []
                for row in cursor.fetchall():
                    salt = bytes(row["salt"])
                    encrypted_material = bytes(row["key_material_encrypted"])
                    key_material = self._decrypt_master(encrypted_material, salt)

                    keys.append(EncryptionKey(
                        key_id=row["key_id"],
                        agent_id=row["agent_id"],
                        key_material=key_material,
                        created_at=datetime.fromisoformat(row["created_at"]),
                        expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                        rotated_at=datetime.fromisoformat(row["rotated_at"]) if row["rotated_at"] else None,
                        algorithm=row["algorithm"],
                        status=row["status"]
                    ))

                return keys
            except Exception as e:
                logger.error(f"Failed to list keys: {e}")
                return []

    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        with self._lock:
            try:
                old_key = self.retrieve_key(key_id)
                if not old_key:
                    return None

                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE encryption_keys
                    SET status = 'rotated', rotated_at = ?
                    WHERE key_id = ?
                """, (datetime.utcnow().isoformat(), key_id))

                new_key_material = secrets.token_bytes(32)
                new_key = EncryptionKey(
                    key_id=str(uuid.uuid4()),
                    agent_id=old_key.agent_id,
                    key_material=new_key_material,
                    created_at=datetime.utcnow(),
                    expires_at=old_key.expires_at,
                    algorithm=old_key.algorithm
                )

                self.store_key(new_key)

                conn.commit()
                return new_key
            except Exception as e:
                logger.error(f"Failed to rotate key: {e}")
                return None


class KMIPKeyStore(BaseKeyStore):
    def __init__(self, host: str, port: int, username: str, password: str,
                 ca_cert: Optional[str] = None, client_cert: Optional[str] = None,
                 client_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ca_cert = ca_cert
        self.client_cert = client_cert
        self.client_key = client_key
        self._client = None
        self._connected = False

    def _connect(self):
        if not KMIP_AVAILABLE:
            raise KeyManagementError("KMIP library not available")
        if self._connected:
            return

        logger.info(f"Connecting to KMIP server at {self.host}:{self.port}")
        self._connected = True

    def _disconnect(self):
        self._connected = False
        self._client = None

    def store_key(self, encryption_key: EncryptionKey) -> bool:
        try:
            self._connect()
            logger.info(f"Storing key {encryption_key.key_id} in KMIP store")
            return True
        except Exception as e:
            logger.error(f"KMIP store key failed: {e}")
            return False

    def retrieve_key(self, key_id: str) -> Optional[EncryptionKey]:
        try:
            self._connect()
            logger.info(f"Retrieving key {key_id} from KMIP store")
            return None
        except Exception as e:
            logger.error(f"KMIP retrieve key failed: {e}")
            return None

    def delete_key(self, key_id: str) -> bool:
        try:
            self._connect()
            logger.info(f"Deleting key {key_id} from KMIP store")
            return True
        except Exception as e:
            logger.error(f"KMIP delete key failed: {e}")
            return False

    def list_keys(self, agent_id: Optional[str] = None) -> List[EncryptionKey]:
        try:
            self._connect()
            logger.info(f"Listing keys from KMIP store for agent {agent_id}")
            return []
        except Exception as e:
            logger.error(f"KMIP list keys failed: {e}")
            return []

    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        try:
            self._connect()
            logger.info(f"Rotating key {key_id} in KMIP store")
            return EncryptionKey(
                key_id=str(uuid.uuid4()),
                agent_id="",
                key_material=secrets.token_bytes(32),
                created_at=datetime.utcnow(),
                algorithm="AES-256"
            )
        except Exception as e:
            logger.error(f"KMIP rotate key failed: {e}")
            return None


class AccessControlManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._connection_pool = {}
        self._lock = threading.RLock()
        self._policy_cache = {}
        self._cache_ttl = 300
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        if thread_id not in self._connection_pool:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connection_pool[thread_id] = conn
        return self._connection_pool[thread_id]

    def _init_database(self):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_policies (
                    policy_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    granted_by TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    expires_at TEXT,
                    conditions TEXT,
                    UNIQUE(agent_id, namespace, role)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role TEXT PRIMARY KEY,
                    permissions TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_policies_agent ON access_policies(agent_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_policies_namespace ON access_policies(namespace)
            """)

            default_permissions = {
                Role.ADMIN: ["read", "write", "delete", "manage", "audit", "encrypt", "decrypt"],
                Role.WRITER: ["read", "write", "encrypt"],
                Role.READER: ["read", "decrypt"],
                Role.AUDITOR: ["read", "audit"],
                Role.NONE: []
            }

            for role, perms in default_permissions.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO role_permissions (role, permissions)
                    VALUES (?, ?)
                """, (role.value, json.dumps(per