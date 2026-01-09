from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.web.routes import router

app = FastAPI(title="ChatGPT Clone Web")

# En local normalmente no hace falta, pero ayuda si separas frontend/backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
