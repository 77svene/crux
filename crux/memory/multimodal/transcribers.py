import os
import io
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Type, Union
from urllib.parse import urlparse

import boto3
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


@dataclass
class MediaMetadata:
    modality: ModalityType
    source_path: Optional[str] = None
    s3_uri: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    frame_count: Optional[int] = None
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    embedding: List[float]
    modality: ModalityType
    model_name: str
    dimensions: int
    normalized: bool = True


@dataclass
class MemoryItem:
    memory_id: str
    modality: ModalityType
    raw_storage_path: str
    extracted_text: Optional[str] = None
    embedding: Optional[EmbeddingResult] = None
    transcription: Optional[TranscriptionResult] = None
    metadata: Optional[MediaMetadata] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def upload(self, data: Union[bytes, BinaryIO], path: str, content_type: str = None) -> str:
        """Upload data and return the storage URI."""
        pass
    
    @abstractmethod
    def download(self, uri: str) -> bytes:
        """Download data from URI and return bytes."""
        pass
    
    @abstractmethod
    def delete(self, uri: str) -> bool:
        """Delete data at URI."""
        pass
    
    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Check if URI exists."""
        pass
    
    @abstractmethod
    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        """Get a presigned URL for temporary access."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str = "./storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload(self, data: Union[bytes, BinaryIO], path: str, content_type: str = None) -> str:
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, bytes):
            full_path.write_bytes(data)
        else:
            full_path.write_bytes(data.read())
        
        return f"local://{path}"
    
    def download(self, uri: str) -> bytes:
        path = self._parse_uri(uri)
        return Path(path).read_bytes()
    
    def delete(self, uri: str) -> bool:
        path = self._parse_uri(uri)
        p = Path(path)
        if p.exists():
            p.unlink()
            return True
        return False
    
    def exists(self, uri: str) -> bool:
        path = self._parse_uri(uri)
        return Path(path).exists()
    
    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        return f"file://{self._parse_uri(uri)}"
    
    def _parse_uri(self, uri: str) -> str:
        if uri.startswith("local://"):
            return str(self.base_path / uri[8:])
        return uri


class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend."""
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        prefix: str = "crux/media"
    ):
        self.bucket = bucket
        self.prefix = prefix
        
        self.client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    
    def upload(self, data: Union[bytes, BinaryIO], path: str, content_type: str = None) -> str:
        full_key = f"{self.prefix}/{path}"
        
        if isinstance(data, BinaryIO):
            data = data.read()
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        
        self.client.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=data,
            **extra_args
        )
        
        return f"s3://{self.bucket}/{full_key}"
    
    def download(self, uri: str) -> bytes:
        key = self._parse_uri(uri)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()
    
    def delete(self, uri: str) -> bool:
        key = self._parse_uri(uri)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
    
    def exists(self, uri: str) -> bool:
        key = self._parse_uri(uri)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
    
    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        key = self._parse_uri(uri)
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in
        )
    
    def _parse_uri(self, uri: str) -> str:
        if uri.startswith("s3://"):
            return uri[5:].split("/", 1)[1]
        return uri


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @abstractmethod
    def embed(self, data: Any, modality: ModalityType) -> EmbeddingResult:
        pass


class CLIPEmbeddingModel(EmbeddingModel):
    """CLIP-based embedding model for images and text."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._model_name = model_name
    
    @property
    def dimensions(self) -> int:
        return 512
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _load_model(self):
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._model = CLIPModel.from_pretrained(self._model_name).to(self._device)
                self._processor = CLIPProcessor.from_pretrained(self._model_name)
                self._model.eval()
            except ImportError:
                logger.warning("transformers not installed, using fallback")
                self._model = self._create_fallback_model()
    
    def _create_fallback_model(self):
        class FallbackModel:
            def __init__(self):
                self.device = "cpu"
            def __call__(self, **kwargs):
                class DummyOutput:
                    last_hidden_state = type('obj', (object,), {'pooler_output': torch.zeros(512)})()
                return DummyOutput()
        return FallbackModel()
    
    def embed(self, data: Any, modality: ModalityType) -> EmbeddingResult:
        self._load_model()
        
        if modality == ModalityType.IMAGE:
            return self._embed_image(data)
        elif modality == ModalityType.TEXT:
            return self._embed_text(data)
        else:
            raise ValueError(f"Unsupported modality for CLIP: {modality}")
    
    def _embed_image(self, image_data: Union[bytes, Image.Image]) -> EmbeddingResult:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            embedding = outputs.cpu().numpy()[0]
        
        embedding = embedding / np.linalg.norm(embedding)
        
        return EmbeddingResult(
            embedding=embedding.tolist(),
            modality=ModalityType.IMAGE,
            model_name=self._model_name,
            dimensions=len(embedding),
            normalized=True
        )
    
    def _embed_text(self, text: str) -> EmbeddingResult:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
            embedding = outputs.cpu().numpy()[0]
        
        embedding = embedding / np.linalg.norm(embedding)
        
        return EmbeddingResult(
            embedding=embedding.tolist(),
            modality=ModalityType.TEXT,
            model_name=self._model_name,
            dimensions=len(embedding),
            normalized=True
        )


class WhisperTranscriber:
    """Audio transcription using OpenAI's Whisper."""
    
    def __init__(
        self,
        model_name: str = "base",
        api_key: str = None,
        api_base: str = "https://api.openai.com/v1",
        use_local: bool = True
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base
        self.use_local = use_local
        self._model = None
    
    def _load_model(self):
        if self._model is None and self.use_local:
            try:
                import whisper
                self._model = whisper.load_model(self.model_name)
            except ImportError:
                logger.warning("whisper not installed, will use API")
                self.use_local = False
    
    def transcribe(
        self,
        audio_data: bytes,
        language: str = None,
        prompt: str = None
    ) -> TranscriptionResult:
        self._load_model()
        
        if self.use_local:
            return self._transcribe_local(audio_data, language, prompt)
        else:
            return self._transcribe_api(audio_data, language, prompt)
    
    def _transcribe_local(
        self,
        audio_data: bytes,
        language: str = None,
        prompt: str = None
    ) -> TranscriptionResult:
        import tempfile
        import whisper
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            result = self._model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
                fp16=torch.cuda.is_available()
            )
            
            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language"),
                confidence=None,
                segments=[
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"]
                    }
                    for seg in result.get("segments", [])
                ],
                metadata={
                    "model": self.model_name,
                    "duration": result.get("duration", 0)
                }
            )
        finally:
            os.unlink(temp_path)
    
    def _transcribe_api(
        self,
        audio_data: bytes,
        language: str = None,
        prompt: str = None
    ) -> TranscriptionResult:
        import base64
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        audio_b64 = base64.b64encode(audio_data).decode()
        
        payload = {
            "model": "whisper-1",
            "file": f"data:audio/wav;base64,{audio_b64}",
            "response_format": "verbose_json"
        }
        
        if language:
            payload["language"] = language
        if prompt:
            payload["prompt"] = prompt
        
        import requests
        response = requests.post(
            f"{self.api_base}/audio/transcriptions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language"),
            segments=result.get("segments", []),
            metadata={
                "model": "whisper-1",
                "duration": result.get("duration", 0),
                "task": result.get("task", "transcribe")
            }
        )


class VideoFrameExtractor:
    """Extract keyframes from video files."""
    
    def __init__(
        self,
        max_frames: int = 16,
        frame_selection: str = "uniform",
        quality: int = 95
    ):
        self.max_frames = max_frames
        self.frame_selection = frame_selection
        self.quality = quality
        self._ffmpeg_available = None
    
    def _check_ffmpeg(self) -> bool:
        if self._ffmpeg_available is None:
            import shutil
            self._ffmpeg_available = shutil.which("ffmpeg") is not None
        return self._ffmpeg_available
    
    def extract_frames(self, video_data: bytes) -> List[Dict[str, Any]]:
        if self._check_ffmpeg():
            return self._extract_with_ffmpeg(video_data)
        else:
            return self._extract_with_cv2(video_data)
    
    def _extract_with_ffmpeg(self, video_data: bytes) -> List[Dict[str, Any]]:
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_data)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    temp_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            duration = float(result.stdout.strip())
            
            frame_times = self._get_frame_times(duration)
            
            frames = []
            for i, timestamp in enumerate(frame_times):
                frame_data = subprocess.run(
                    [
                        "ffmpeg",
                        "-ss", str(timestamp),
                        "-i", temp_path,
                        "-vframes", "1",
                        "-q:v", str(2),
                        "-f", "image2pipe",
                        "-vcodec", "png",
                        "-"
                    ],
                    capture_output=True,
                    timeout=10
                )
                
                if frame_data.returncode == 0 and frame_data.stdout:
                    img = Image.open(io.BytesIO(frame_data.stdout))
                    frames.append({
                        "frame_index": i,
                        "timestamp": timestamp,
                        "image": img,
                        "image_bytes": frame_data.stdout
                    })
            
            return frames
        finally:
            os.unlink(temp_path)
    
    def _extract_with_cv2(self, video_data: bytes) -> List[Dict[str, Any]]:
        try:
            import cv2
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_data)
                temp_path = f.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                frame_times = self._get_frame_times