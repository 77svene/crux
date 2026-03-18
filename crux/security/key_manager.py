import os
import json
import uuid
import hashlib
import hmac
import base64
import struct
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from functools import wraps
from abc import ABC, abstractmethod

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.x509.oid import NameOID

try:
    from sqlalchemy import Column, String, DateTime, Boolean, Integer, LargeBinary, Text, ForeignKey, JSON, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = object

logger = logging.getLogger(__name__)


class KeyStatus(Enum):
    ACTIVE = "active"
    COMPROMISED = "compromised"
    REVOKED = "revoked"
    DESTROYED = "destroyed"
    EXPIRED = "expired"


class AccessLevel(Enum):
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3


class ComplianceAction(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXPORT = "export"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    KEY_GENERATE = "key_generate"
    KEY_ACCESS = "key_access"
    ACCESS_DENIED = "access_denied"


@dataclass
class AuditEntry:
    timestamp: datetime
    agent_id: str
    action: ComplianceAction
    resource_id: Optional[str]
    success: bool
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AgentKey:
    agent_id: str
    key_id: str
    encrypted_key: bytes
    key_check_value: bytes
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_accessed: Optional[datetime]
    rotation_count: int
    metadata: Dict[str, Any]


@dataclass
class AccessPolicy:
    policy_id: str
    agent_id: str
    namespace: str
    access_level: AccessLevel
    conditions: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime]


class KMIPKeyStore(ABC):
    @abstractmethod
    def generate_key(self, key_id: str, algorithm: str = "AES-256") -> bytes:
        pass
    
    @abstractmethod
    def get_key(self, key_id: str) -> Optional[bytes]:
        pass
    
    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        pass


class InMemoryKeyStore(KMIPKeyStore):
    def __init__(self):
        self._keys: Dict[str, bytes] = {}
        self._lock = threading.RLock()
    
    def generate_key(self, key_id: str, algorithm: str = "AES-256") -> bytes:
        with self._lock:
            if algorithm == "AES-256":
                key = os.urandom(32)
            elif algorithm == "AES-128":
                key = os.urandom(16)
            elif algorithm == "AES-512":
                key = os.urandom(64)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            self._keys[key_id] = key
            return key
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        with self._lock:
            return self._keys.get(key_id)
    
    def delete_key(self, key_id: str) -> bool:
        with self._lock:
            if key_id in self._keys:
                del self._keys[key_id]
                return True
            return False
    
    def list_keys(self) -> List[str]:
        with self._lock:
            return list(self._keys.keys())


class SQLAlchemyKeyStore(KMIPKeyStore):
    def __init__(self, connection_string: str, master_key: Optional[bytes] = None):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for SQLAlchemyKeyStore")
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
        if master_key is None:
            master_key = os.environ.get("MEM0_MASTER_KEY", "").encode()
            if not master_key:
                import secrets
                master_key = secrets.token_bytes(32)
        
        self._master_key = master_key
        self._aesgcm = AESGCM(master_key[:32])
        self._lock = threading.RLock()
        
        Base.metadata.create_all(self.engine)
    
    def _encrypt_key_material(self, key_material: bytes) -> bytes:
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, key_material, None)
        return nonce + ciphertext
    
    def _decrypt_key_material(self, encrypted: bytes) -> bytes:
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        return self._aesgcm.decrypt(nonce, ciphertext, None)
    
    def generate_key(self, key_id: str, algorithm: str = "AES-256") -> bytes:
        with self._lock:
            session = self.Session()
            try:
                from sqlalchemy import Column, String, LargeBinary, DateTime, Integer
                
                class KeyMaterial(Base):
                    __tablename__ = "key_material"
                    
                    key_id = Column(String(255), primary_key=True)
                    encrypted_material = Column(LargeBinary)
                    algorithm = Column(String(50))
                    created_at = Column(DateTime, default=datetime.utcnow)
                    rotation_count = Column(Integer, default=0)
                
                if algorithm == "AES-256":
                    key = os.urandom(32)
                elif algorithm == "AES-128":
                    key = os.urandom(16)
                elif algorithm == "AES-512":
                    key = os.urandom(64)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                encrypted = self._encrypt_key_material(key)
                
                km = KeyMaterial(
                    key_id=key_id,
                    encrypted_material=encrypted,
                    algorithm=algorithm
                )
                session.merge(km)
                session.commit()
                
                return key
            finally:
                session.close()
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        with self._lock:
            session = self.Session()
            try:
                from sqlalchemy import Column, String, LargeBinary, DateTime, Integer
                
                class KeyMaterial(Base):
                    __tablename__ = "key_material"
                    
                    key_id = Column(String(255), primary_key=True)
                    encrypted_material = Column(LargeBinary)
                    algorithm = Column(String(50))
                    created_at = Column(DateTime, default=datetime.utcnow)
                    rotation_count = Column(Integer, default=0)
                
                km = session.query(KeyMaterial).filter_by(key_id=key_id).first()
                if km:
                    return self._decrypt_key_material(km.encrypted_material)
                return None
            finally:
                session.close()
    
    def delete_key(self, key_id: str) -> bool:
        with self._lock:
            session = self.Session()
            try:
                from sqlalchemy import Column, String, LargeBinary, DateTime, Integer
                
                class KeyMaterial(Base):
                    __tablename__ = "key_material"
                    
                    key_id = Column(String(255), primary_key=True)
                    encrypted_material = Column(LargeBinary)
                    algorithm = Column(String(50))
                    created_at = Column(DateTime, default=datetime.utcnow)
                    rotation_count = Column(Integer, default=0)
                
                km = session.query(KeyMaterial).filter_by(key_id=key_id).first()
                if km:
                    session.delete(km)
                    session.commit()
                    return True
                return False
            finally:
                session.close()


class KeyManager:
    def __init__(
        self,
        keystore: Optional[KMIPKeyStore] = None,
        database_url: Optional[str] = None,
        master_key: Optional[bytes] = None,
        key_expiry_days: int = 90,
        enable_auto_rotation: bool = True,
        rotation_interval_days: int = 30,
    ):
        if keystore:
            self._keystore = keystore
        elif database_url:
            self._keystore = SQLAlchemyKeyStore(database_url, master_key)
        else:
            self._keystore = InMemoryKeyStore()
        
        self._key_expiry_days = key_expiry_days
        self._enable_auto_rotation = enable_auto_rotation
        self._rotation_interval_days = rotation_interval_days
        self._master_key = master_key or os.environ.get("MEM0_MASTER_KEY", "").encode() or os.urandom(32)
        self._lock = threading.RLock()
        self._key_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self._cache_ttl_seconds = 300
        
        self._initialize_tables()
    
    def _initialize_tables(self):
        if SQLALCHEMY_AVAILABLE and isinstance(self._keystore, SQLAlchemyKeyStore):
            return
        
        if SQLALCHEMY_AVAILABLE:
            try:
                engine = create_engine("sqlite:///:memory:")
                Base.metadata.create_all(engine)
            except Exception:
                pass
    
    def _derive_agent_key(self, agent_id: str, master_key: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f"crux-agent-key-{agent_id}".encode(),
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(master_key)
    
    def _compute_key_check_value(self, key: bytes) -> bytes:
        return hashlib.sha256(key[:16]).digest()[:8]
    
    def _get_cache_key(self, agent_id: str) -> str:
        return hashlib.sha256(f"{agent_id}".encode()).hexdigest()
    
    def _is_key_cached(self, cache_key: str) -> bool:
        if cache_key not in self._key_cache:
            return False
        _, cached_time = self._key_cache[cache_key]
        return (datetime.utcnow() - cached_time).total_seconds() < self._cache_ttl_seconds
    
    def _cache_key_material(self, agent_id: str, key: bytes):
        cache_key = self._get_cache_key(agent_id)
        self._key_cache[cache_key] = (key, datetime.utcnow())
    
    def _get_cached_key(self, agent_id: str) -> Optional[bytes]:
        cache_key = self._get_cache_key(agent_id)
        if self._is_key_cached(cache_key):
            return self._key_cache[cache_key][0]
        return None
    
    def get_or_create_agent_key(
        self,
        agent_id: str,
        force_rotation: bool = False
    ) -> AgentKey:
        with self._lock:
            cached = self._get_cached_key(agent_id)
            if cached and not force_rotation:
                return AgentKey(
                    agent_id=agent_id,
                    key_id=self._generate_key_id(agent_id),
                    encrypted_key=cached,
                    key_check_value=self._compute_key_check_value(cached),
                    status=KeyStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=self._key_expiry_days),
                    last_accessed=datetime.utcnow(),
                    rotation_count=0,
                    metadata={}
                )
            
            key_id = self._generate_key_id(agent_id)
            existing_key = self._get_stored_agent_key(agent_id)
            
            if existing_key and not force_rotation:
                if existing_key.status == KeyStatus.ACTIVE:
                    if not self._is_key_expired(existing_key):
                        self._cache_key_material(agent_id, existing_key.encrypted_key)
                        return existing_key
            
            new_key = self._generate_agent_key(agent_id)
            
            agent_key = AgentKey(
                agent_id=agent_id,
                key_id=key_id,
                encrypted_key=new_key,
                key_check_value=self._compute_key_check_value(new_key),
                status=KeyStatus.ACTIVE,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=self._key_expiry_days),
                last_accessed=datetime.utcnow(),
                rotation_count=(existing_key.rotation_count + 1) if existing_key else 1,
                metadata={"algorithm": "AES-256-GCM"}
            )
            
            self._store_agent_key(agent_key)
            self._cache_key_material(agent_id, new_key)
            
            return agent_key
    
    def _generate_key_id(self, agent_id: str) -> str:
        timestamp = int(time.time())
        raw = f"{agent_id}-{timestamp}-{uuid.uuid4()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
    
    def _generate_agent_key(self, agent_id: str) -> bytes:
        return self._keystore.generate_key(f"agent-{agent_id}", "AES-256")
    
    def _is_key_expired(self, agent_key: AgentKey) -> bool:
        if agent_key.expires_at is None:
            return False
        return datetime.utcnow() > agent_key.expires_at
    
    def _get_stored_agent_key(self, agent_id: str) -> Optional[AgentKey]:
        return None
    
    def _store_agent_key(self, agent_key: AgentKey):
        pass
    
    def revoke_agent_key(self, agent_id: str, reason: str = "") -> bool:
        with self._lock:
            cache_key = self._get_cache_key(agent_id)
            if cache_key in self._key_cache:
                del self._key_cache[cache_key]
            
            logger.warning(f"Key revoked for agent {agent_id}: {reason}")
            return True
    
    def destroy_agent_key(self, agent_id: str) -> bool:
        with self._lock:
            cache_key = self._get_cache_key(agent_id)
            if cache_key in self._key_cache:
                del self._key_cache[cache_key]
            
            key_id = f"agent-{agent_id}"
            result = self._keystore.delete_key(key_id)
            
            logger.critical(f"Key DESTROYED for agent {agent_id} - GDPR erasure complete")
            return result
    
    def verify_key(self, agent_id: str, key_check_value: bytes) -> bool:
        agent_key = self.get_or_create_agent_key(agent_id)
        return hmac.compare_digest(agent_key.key_check_value, key_check_value)
    
    def list_agent_keys(self) -> List[str]:
        return self._keystore.list_keys()
    
    def get_key_status(self, agent_id: str) -> KeyStatus:
        agent_key = self._get_stored_agent_key(agent_id)
        if agent_key is None:
            return KeyStatus.ACTIVE
        return agent_key.status


class EncryptionManager:
    def __init__(
        self,
        key_manager: Optional[KeyManager] = None,
        aead_mode: bool = True,
        enable_key_check: bool = True,
    ):
        self._key_manager = key_manager or KeyManager()
        self._aead_mode = aead_mode
        self._enable_key_check = enable_key_check
        self._lock = threading.RLock()
        self._nonce_counter: Dict[str, int] = {}
    
    def _get_nonce(self, agent_id: str) -> bytes:
        with self._lock:
            if agent_id not in self._nonce_counter:
                self._nonce_counter[agent_id] = 0
            
            counter = self._nonce_counter[agent_id]
            self._nonce_counter[agent_id] = (counter + 1) % (2**64)
            
            if self._aead_mode:
                return os.urandom(12)
            else:
                counter_bytes = struct.pack('<Q', counter)
                return counter_bytes.ljust(12, b'\x00')
    
    def _pad_nonce(self, nonce: bytes) -> bytes:
        if len(nonce) < 12:
            return nonce.ljust(12, b'\x00')
        elif len(nonce) > 12:
            return nonce[:12]
        return nonce
    
    def encrypt(
        self,
        data: str,
        agent_id: str,
        additional_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        if not data:
            return {"ciphertext": b"", "nonce": b"", "key_id": ""}
        
        agent_key = self._key_manager.get_or_create_agent_key(agent_id)
        key = agent_key.encrypted_key
        nonce = self._get_nonce(agent_id)
        
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        if self._aead_mode:
            aesgcm = AESGCM(key)
            if additional_data:
                ciphertext = aesgcm.encrypt(nonce, data_bytes, additional_data)
            else:
                ciphertext = aesgcm.encrypt(nonce, data_bytes, agent_id.encode())
        else:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data_bytes)