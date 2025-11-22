from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import base64
import os
from config import engine
from sqlalchemy import text
import tables.users as user_tables
import tables.recordings as recordings_tables
import routes.users as user_routes
import routes.recordings as recordings_routes

user_tables.Base.metadata.create_all(bind=engine)
recordings_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")


@app.on_event("startup")
def ensure_recordings_schema():
    """Ensure the `care_recipient_id` column exists on startup to avoid runtime SQL errors.

    This is a defensive, idempotent migration useful during development. For production
    deployments prefer real migrations (alembic).
    """
    try:
        sql_add_col = "ALTER TABLE recordings ADD COLUMN IF NOT EXISTS care_recipient_id integer;"
        # Try to add an FK constraint only if possible; if the referenced table doesn't
        # exist yet, skip the FK creation gracefully.
        sql_add_fk = '''DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name='recordings' AND tc.constraint_type='FOREIGN KEY' AND kcu.column_name='care_recipient_id'
            ) THEN
                BEGIN
                    ALTER TABLE recordings ADD CONSTRAINT recordings_care_recipient_fk FOREIGN KEY (care_recipient_id) REFERENCES care_recipients(id) ON DELETE SET NULL;
                EXCEPTION WHEN undefined_table THEN
                    -- referenced table missing; skip adding FK for now
                    RAISE NOTICE 'care_recipients table missing; skipping FK creation';
                END;
            END IF;
        END$$;'''
        with engine.begin() as conn:
            conn.execute(text(sql_add_col))
            conn.execute(text(sql_add_fk))
        print("Startup schema check: ensured recordings.care_recipient_id exists (FK added if possible).")
    except Exception as e:
        print("Startup schema check failed:", e)

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




