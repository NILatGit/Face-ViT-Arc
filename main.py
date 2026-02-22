import modal
from app.config import app, image, vol, api_secret
from app.engine import FaceEngine  # noqa: F401 - registers @app.cls with Modal


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[api_secret],
    min_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    from app.server import build_app

    return build_app()
