"""Fixtures compartidos para la suite de pruebas."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from tasador.config import Settings
from tasador.main import create_app


def make_test_image_bytes(
    color: tuple[int, int, int] = (210, 200, 185),
    size: tuple[int, int] = (320, 240),
    fmt: str = "JPEG",
) -> bytes:
    """Genera una imagen sintética válida en memoria con detalle arquitectónico simple."""
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, size[0] - 40, size[1] - 40], outline=(80, 80, 90), width=4)
    draw.rectangle([90, 70, 180, 170], fill=(245, 248, 255), outline=(60, 60, 70), width=2)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def demo_client() -> TestClient:
    """Cliente de pruebas FastAPI forzado en Modo Demo Local (sin llamadas externas)."""
    settings = Settings(_env_file=None, openrouter_api_key="")
    app = create_app(settings_override=settings)
    with TestClient(app) as client:
        yield client
