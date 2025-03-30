from fastapi import FastAPI
from app.routes import router
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# List of allowed origins (replace with your frontend URL if different)
origins = [
    "http://localhost:8081",  # frontend URL
]

app = FastAPI(title="SandalQuest API", version="0.0.1",
              description="API for SandalQuest project")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include router for API routes
app.include_router(router)

# Mounting static files url: /static/outputs
app.mount("/static/outputs", StaticFiles(directory="app/static/outputs"), name="outputs")

@app.get("/")
def read_root():
    return {"message": "Welcome to SandalQuest API"}
