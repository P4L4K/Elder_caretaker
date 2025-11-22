from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import base64
import os
from config import engine
import tables.users as user_tables
import tables.recordings as recordings_tables
import routes.users as user_routes
import routes.recordings as recordings_routes

user_tables.Base.metadata.create_all(bind=engine)
recordings_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

# Serve a static folder (optional) so files like a favicon can be served
static_path = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_path):
    try:
        os.makedirs(static_path, exist_ok=True)
    except Exception:
        pass
app.mount("/static", StaticFiles(directory=static_path), name="static")
# Also mount the repository-level `model` directory (serves model.json, metadata.json, weights.bin)
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
if os.path.exists(model_dir):
    app.mount("/model", StaticFiles(directory=model_dir), name="model")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # Allow all origins for development to avoid CORS issues with the frontend static server.
    # Note: when `allow_origins` is ['*'] you MUST set `allow_credentials=False`.
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        # Explicitly allow the Authorization header and common content headers to satisfy preflight
        allow_headers=["Authorization", "Content-Type", "Accept"],
        expose_headers=["Content-Disposition"]
)

@app.get("/")
async def root():
    return {"message": "Welcome to CareTaker API", "status": "active"}


# Return a small in-memory favicon to avoid 404s when browsers request it.
# If a real favicon file exists in `backend/static/favicon.ico` it will be served instead.
@app.get('/favicon.ico')
async def favicon():
    ico_path = os.path.join(static_path, 'favicon.ico')
    if os.path.exists(ico_path):
        return FileResponse(ico_path)
    # 1x1 transparent PNG (base64) returned as image/png for simplicity
    png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
    return Response(content=base64.b64decode(png_b64), media_type='image/png')

# Include routers without prefix since we're handling it in the router
app.include_router(user_routes.router)
app.include_router(recordings_routes.router)




