from __future__ import annotations

import argparse
import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

try:
    from .schemas import (
        HealthzResponse,
        Token2RewardsRequest,
        Token2RewardsResponse,
        Token2WavRequest,
        Token2WerRequest,
        Token2WerResponse,
    )
    from .service import ServiceConfig, TokenService
except ImportError:  # pragma: no cover
    from schemas import (
        HealthzResponse,
        Token2RewardsRequest,
        Token2RewardsResponse,
        Token2WavRequest,
        Token2WerRequest,
        Token2WerResponse,
    )
    from service import ServiceConfig, TokenService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> ServiceConfig:
    parser = argparse.ArgumentParser(description="Dockerized CosyVoice token services")
    parser.add_argument("--cosyvoice-model-dir", required=True)
    parser.add_argument("--asr-model-id", default="iic/SenseVoiceSmall")
    parser.add_argument("--spk-embed-model-id", default="")
    parser.add_argument("--dnsmos-model-id", default="")
    parser.add_argument("--prompt-dataset-id", default="yuekai/aishell")
    parser.add_argument("--prompt-dataset-split", default="test")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return ServiceConfig(
        cosyvoice_model_dir=args.cosyvoice_model_dir,
        asr_model_id=args.asr_model_id,
        spk_embed_model_id=args.spk_embed_model_id,
        dnsmos_model_id=args.dnsmos_model_id,
        prompt_dataset_id=args.prompt_dataset_id,
        prompt_dataset_split=args.prompt_dataset_split,
        host=args.host,
        port=args.port,
        device=args.device,
        fp16=args.fp16,
        max_audio_seconds=args.max_audio_seconds,
        seed=args.seed,
    )
def create_app(config: ServiceConfig) -> FastAPI:
    service = TokenService(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting token services with model_dir=%s", config.cosyvoice_model_dir)
        service.load()
        app.state.token_service = service
        yield

    app = FastAPI(title="CosyVoice Token Services", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz():
        return app.state.token_service.healthz()

    @app.post("/token2wav")
    async def token2wav(request: Token2WavRequest):
        request_id = uuid.uuid4().hex[:8]
        logger.info("request_id=%s endpoint=/token2wav", request_id)
        try:
            wav_bytes = app.state.token_service.token2wav(request)
            return Response(content=wav_bytes, media_type="audio/wav")
        except ValueError as exc:
            logger.warning("request_id=%s bad_request=%s", request_id, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("request_id=%s decode_failed", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/token2wer", response_model=Token2WerResponse)
    async def token2wer(request: Token2WerRequest):
        request_id = uuid.uuid4().hex[:8]
        logger.info("request_id=%s endpoint=/token2wer", request_id)
        try:
            return app.state.token_service.token2wer(request)
        except ValueError as exc:
            logger.warning("request_id=%s bad_request=%s", request_id, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("request_id=%s token2wer_failed", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/token2rewards", response_model=Token2RewardsResponse)
    async def token2rewards(request: Token2RewardsRequest):
        request_id = uuid.uuid4().hex[:8]
        logger.info("request_id=%s endpoint=/token2rewards", request_id)
        try:
            return app.state.token_service.token2rewards(request)
        except ValueError as exc:
            logger.warning("request_id=%s bad_request=%s", request_id, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("request_id=%s token2rewards_failed", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


if __name__ == "__main__":
    config = parse_args()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
