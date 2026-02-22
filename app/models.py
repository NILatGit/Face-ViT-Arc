from pydantic import BaseModel
from typing import Optional


class VerifyResponse(BaseModel):
    match: bool
    confidence: float
    log_id: int


class IdentifyResponse(BaseModel):
    faiss_idx: int
    name: str
    confidence: float
    log_id: int


class RegisterResponse(BaseModel):
    success: bool
    faiss_idx: Optional[int]
    message: str


class LogEntry(BaseModel):
    id: int
    operation: str
    result: str
    confidence: float
    feedback: Optional[str]
    created_at: str


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: str


class HealthResponse(BaseModel):
    status: str
    registered_faces: int
