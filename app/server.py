import os
from fastapi import FastAPI, UploadFile, File, Form, Security, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_api_key
from app.config import MAX_IMAGE_BYTES, VERIFY_THRESHOLD
from app.db import Database
from app.engine import FaceEngine
from app.models import (
    FeedbackRequest,
    HealthResponse,
    IdentifyResponse,
    LogEntry,
    RegisterResponse,
    VerifyResponse,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _read_upload(upload: UploadFile) -> bytes:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {upload.content_type}",
        )
    data = upload.file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 10 MB limit",
        )
    return data


def build_app() -> FastAPI:
    web = FastAPI(title="Face Recognition API")

    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db = Database()
    engine = FaceEngine()

    @web.get("/api/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok",
            registered_faces=db.count_identities(),
        )

    @web.post(
        "/api/verify",
        response_model=VerifyResponse,
        dependencies=[Security(verify_api_key)],
    )
    async def verify_endpoint(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
    ) -> VerifyResponse:
        b1 = _read_upload(file1)
        b2 = _read_upload(file2)

        score = engine.verify.remote(b1, b2)
        if score is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Face not detected in one or both images",
            )

        match = score >= VERIFY_THRESHOLD
        log_id = db.insert_log(
            operation="verify",
            result="match" if match else "no_match",
            confidence=score,
        )
        return VerifyResponse(match=match, confidence=score, log_id=log_id)

    @web.post(
        "/api/identify",
        response_model=IdentifyResponse,
        dependencies=[Security(verify_api_key)],
    )
    async def identify_endpoint(
        file: UploadFile = File(...),
    ) -> IdentifyResponse:
        b = _read_upload(file)

        result = engine.identify.remote(b)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No face detected",
            )

        faiss_idx, score = result
        name = db.get_identity(faiss_idx)
        if name is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Identity not found in database",
            )

        log_id = db.insert_log(operation="identify", result=name, confidence=score)
        return IdentifyResponse(
            faiss_idx=faiss_idx, name=name, confidence=score, log_id=log_id
        )

    @web.post(
        "/api/register",
        response_model=RegisterResponse,
        dependencies=[Security(verify_api_key)],
    )
    async def register_endpoint(
        file: UploadFile = File(...),
        name: str = Form(...),
    ) -> RegisterResponse:
        b = _read_upload(file)

        faiss_idx = db.count_identities()
        success = engine.register.remote(b, faiss_idx)
        if not success:
            return RegisterResponse(
                success=False,
                faiss_idx=None,
                message="No face detected in the provided image",
            )

        db.insert_identity(faiss_idx=faiss_idx, name=name)
        return RegisterResponse(
            success=True,
            faiss_idx=faiss_idx,
            message=f"Registered '{name}' at index {faiss_idx}",
        )

    @web.get(
        "/api/history",
        response_model=list[LogEntry],
        dependencies=[Security(verify_api_key)],
    )
    def history_endpoint() -> list[LogEntry]:
        return db.get_recent_logs(limit=50)

    @web.post(
        "/api/feedback",
        dependencies=[Security(verify_api_key)],
    )
    def feedback_endpoint(data: FeedbackRequest) -> dict:
        updated = db.update_feedback(log_id=data.log_id, feedback=data.feedback)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No log entry with id {data.log_id}",
            )
        return {"status": "ok"}

    return web
