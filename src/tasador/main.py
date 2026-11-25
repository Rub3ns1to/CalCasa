"""Punto de entrada FastAPI para el Tasador Inmobiliario Multimodal."""

import logging
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from tasador import __version__
from tasador.config import Settings, get_settings
from tasador.ml.features import PROPERTY_TYPES
from tasador.ml.predictor import TabularPricePredictor
from tasador.schemas import AppraisalResponse, PropertyInput
from tasador.services.appraisal_service import AppraisalService
from tasador.services.pdf_report import generate_appraisal_pdf
from tasador.vision.openrouter_client import OpenRouterVisionClient, VisionClientError


def create_app(settings_override: Settings | None = None) -> FastAPI:
    """Construye y configura la instancia principal de FastAPI."""
    settings = settings_override or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        predictor = TabularPricePredictor(
            model_path=settings.model_path,
            metadata_path=settings.metadata_path,
            comparables_path=settings.comparables_path,
        )
        predictor.load()
        vision_client = OpenRouterVisionClient(settings=settings)
        appraisal_service = AppraisalService(
            settings=settings,
            predictor=predictor,
            vision_client=vision_client,
        )
        application.state.settings = settings
        application.state.predictor = predictor
        application.state.appraisal_service = appraisal_service
        yield

    app = FastAPI(
        title="CalCasa · Tasador Inmobiliario Multimodal (Madrid)",
        description=(
            "Estimación del valor de viviendas combinando Machine Learning tabular (XGBoost) "
            "con auditoría visual multimodal vía OpenRouter."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    class NoCacheStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope: Any) -> Response:
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

    if settings.static_dir.exists():
        app.mount("/static", NoCacheStaticFiles(directory=str(settings.static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        index_path = settings.static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Interfaz web no encontrada.")
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/api/health")
    async def health_check(request: Request) -> dict[str, Any]:
        predictor: TabularPricePredictor = request.app.state.predictor
        cfg: Settings = request.app.state.settings
        target = cfg.resolve_vision_target()
        return {
            "status": "ok" if predictor.is_loaded else "model_missing",
            "model_loaded": predictor.is_loaded,
            "vision_mode": target.provider,
            "vision_model": target.model,
            "openrouter_model": target.model,
            "byok_supported": True,
            "version": __version__,
        }

    @app.get("/api/metadata")
    async def get_metadata(request: Request) -> dict[str, Any]:
        predictor: TabularPricePredictor = request.app.state.predictor
        cfg: Settings = request.app.state.settings
        return {
            "districts": predictor.get_available_districts(),
            "property_types": PROPERTY_TYPES,
            "model": predictor.get_metadata(),
            "limits": {
                "min_images": cfg.min_images,
                "max_images": cfg.max_images,
                "max_image_size_mb": cfg.max_image_size_mb,
                "max_visual_adjustment_pct": cfg.max_visual_adjustment_pct,
            },
        }

    @app.post("/api/appraise", response_model=AppraisalResponse)
    async def appraise_property(
        request: Request,
        barrio: str = Form(...),
        micro_barrio: str | None = Form(None),
        tipo_vivienda: str = Form("Piso"),
        metros_cuadrados: float = Form(...),
        habitaciones: int = Form(...),
        banos: int = Form(1),
        planta: int = Form(1),
        ascensor: bool = Form(True),
        exterior: bool = Form(True),
        terraza: bool = Form(False),
        garaje: bool = Form(False),
        aire_acondicionado: bool = Form(False),
        trastero: bool = Form(False),
        fotos: list[UploadFile] = File(...),
        x_vision_api_key: str | None = Header(default=None, alias="X-Vision-Api-Key"),
        x_vision_provider: str = Header(default="auto", alias="X-Vision-Provider"),
        x_vision_model: str | None = Header(default=None, alias="X-Vision-Model"),
    ) -> AppraisalResponse:
        try:
            prop = PropertyInput(
                barrio=barrio,
                micro_barrio=micro_barrio,
                tipo_vivienda=tipo_vivienda,
                metros_cuadrados=metros_cuadrados,
                habitaciones=habitaciones,
                banos=banos,
                planta=planta,
                ascensor=ascensor,
                exterior=exterior,
                terraza=terraza,
                garaje=garaje,
                aire_acondicionado=aire_acondicionado,
                trastero=trastero,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        uploaded_files: list[tuple[str, bytes]] = []
        for foto in fotos:
            content = await foto.read()
            uploaded_files.append((foto.filename or "imagen.jpg", content))

        service: AppraisalService = request.app.state.appraisal_service
        try:
            return await service.appraise(
                prop=prop,
                uploaded_files=uploaded_files,
                byok_api_key=x_vision_api_key,
                byok_provider=x_vision_provider,
                byok_model=x_vision_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VisionClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/report/pdf")
    async def export_appraisal_pdf(appraisal: AppraisalResponse) -> Response:
        pdf_bytes = generate_appraisal_pdf(appraisal)
        zone = appraisal.propiedad.micro_barrio or appraisal.propiedad.barrio
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", zone).strip("_").lower() or "madrid"
        filename = f"calcasa_{slug}_{int(appraisal.propiedad.metros_cuadrados)}m2.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.exception_handler(Exception)
    async def fallback_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception("Error interno no controlado: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno al procesar la tasación."},
        )

    return app


app = create_app()
