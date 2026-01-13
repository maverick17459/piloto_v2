
# src/main.py

from dotenv import load_dotenv
load_dotenv()

import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.web.routes import router
from src.observability.logger import get_logger

app = FastAPI(title="ChatGPT Clone Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
    log = get_logger(trace_id)


    t0 = time.time()
    log.info(f"request.start method={request.method} path={request.url.path}")

    response = await call_next(request)

    ms = int((time.time() - t0) * 1000)
    log.info(f"request.end status={response.status_code} duration_ms={ms}")

    response.headers["X-Trace-Id"] = trace_id
    return response

app.include_router(router)

