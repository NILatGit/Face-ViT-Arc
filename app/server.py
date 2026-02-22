"""
FastAPI application factory and all HTTP route definitions.

This module is the HTTP boundary of the application. It handles:
  - Request validation (file type, size)
  - Authentication enforcement (via app/auth.py dependency)
  - Delegating inference work to the GPU container (app/engine.py)
  - Delegating persistence to the database layer (app/db.py)
  - Returning typed JSON responses (app/models.py schemas)

build_app() is a factory function rather than a module-level FastAPI instance
because Modal calls it inside the container at startup (see main.py). This
ensures the Database connection and FaceEngine handle are created fresh each
time a new CPU container boots, rather than at import time on the developer's
machine.
"""

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

# File types accepted by all upload endpoints. The check is against the
# Content-Type header of the multipart file part, not the file extension.
# Browsers and HTTP clients set this automatically when building a FormData
# upload. "image/jpg" is included alongside "image/jpeg" because some clients
# send the non-standard variant.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def _read_upload(upload: UploadFile) -> bytes:
    """
    Validate and read an uploaded image file into bytes.

    This is called at the top of every endpoint that accepts an image before
    passing the bytes to the engine. It enforces two hard limits:
      1. Content type must be one of ALLOWED_CONTENT_TYPES.
      2. File size must not exceed MAX_IMAGE_BYTES (config.py, default 10 MB).

    Raising HTTPException here short-circuits the route handler - the engine
    is never called and no log entry is written for rejected uploads.
    """
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
    """
    Construct and configure the FastAPI application.

    Called once per CPU container startup from main.py's fastapi_app function.
    Instantiates the Database and FaceEngine once and closes over them in all
    route handlers - this means every request in the same container reuses the
    same DB connection and the same engine handle without reconnecting.
    """
    web = FastAPI(title="Face Recognition API")

    # Read allowed origins from the environment. If ALLOWED_ORIGINS is not
    # set (e.g. during local modal serve without the env var in the secret),
    # defaults to "*" which permits all origins. Split on comma to support
    # multiple origins in a single env var value.
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Instantiated once per container. All route handlers close over these
    # two objects. db holds a persistent SQLite connection; engine is a Modal
    # Class handle that proxies .remote() calls to the GPU container.
    db = Database()
    engine = FaceEngine()

    @web.get("/api/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        """
        Public liveness check. No authentication required.

        Returns the server status and the current count of registered faces
        from the identities table. Useful for confirming the container is
        running and the database is reachable before making inference calls.
        Does not call the GPU engine, so it is always fast regardless of
        whether the GPU container is warm.
        """
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
        """
        1:1 face verification. Determines whether two images show the same person.

        Both images are sent to the GPU container in a single .remote() call.
        The engine embeds both faces and returns their cosine similarity score.
        The score is compared against VERIFY_THRESHOLD (default 0.6) to
        produce a boolean match result.

        Returns 422 if either image contains no detectable face.
        """
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
        """
        1:N face identification. Finds the closest registered identity to a probe.

        The probe image is sent to the GPU container, which embeds the face and
        performs a nearest-neighbour search over the entire FAISS index. The
        integer faiss_idx of the best match is returned and resolved to a name
        via the identities table.

        Returns 422 if no face is detected in the probe image.
        Returns 404 if the FAISS index returns an ID that has no corresponding
        row in the identities table (indicates an index/DB sync issue).

        Note: this endpoint does not apply a confidence threshold - it always
        returns the closest match regardless of score. If you want to reject
        low-confidence identifications, check the confidence value in the
        response and treat scores below your chosen threshold as "unknown"
        on the client side, or add a threshold check before the db.get_identity
        call and return a 404 when the score is too low.
        """
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
        """
        Register a new identity by associating a name with a face embedding.

        The next available faiss_idx is determined by counting current rows in
        the identities table. The engine inserts the embedding at that ID in the
        FAISS index; if successful, the DB record is written with the same ID.

        The two writes (FAISS via engine.register.remote and SQLite via
        db.insert_identity) are not wrapped in a distributed transaction. If the
        DB write fails after the FAISS write succeeds, the index will contain an
        orphaned vector. To recover, either re-register the face (which creates a
        new ID) or manually delete the orphaned FAISS vector using engine.remove().

        Returns a RegisterResponse with success=False (HTTP 200, not 422) if no
        face is detected. This is intentional - the request was valid, the image
        just didn't contain a usable face. The client can prompt the user to
        resubmit without treating it as an error.
        """
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
        """
        Return the 50 most recent audit log entries, newest first.
        """
        return db.get_recent_logs(limit=50)

    @web.post(
        "/api/feedback",
        dependencies=[Security(verify_api_key)],
    )
    def feedback_endpoint(data: FeedbackRequest) -> dict:
        """
        Attach a feedback label to an existing log entry.

        Allows the client to mark a recognition result as correct or incorrect
        after the fact. The feedback string is free-form; the API does not
        enforce a vocabulary. Typical values: "correct", "incorrect".

        The log_id in the request body must match the log_id returned by the
        verify, identify, or register endpoint that produced the entry.

        Returns 404 if no log entry with the given id exists.
        """
        updated = db.update_feedback(log_id=data.log_id, feedback=data.feedback)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No log entry with id {data.log_id}",
            )
        return {"status": "ok"}

    return web
