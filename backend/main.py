from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import base64
import os
from config import engine
from sqlalchemy import text
import tables.users as user_tables
import tables.recordings as recordings_tables
import tables.medical_reports as med_reports_tables
import routes.users as user_routes
import routes.recordings as recordings_routes
import routes.recipients as recipients_routes
import routes.emotion as emotion_routes

from routes import elderly

user_tables.Base.metadata.create_all(bind=engine)
recordings_tables.Base.metadata.create_all(bind=engine)
med_reports_tables.Base.metadata.create_all(bind=engine)

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
        # Ensure care_recipients.report_summary column exists (idempotent)
        try:
            sql_add_summary = "ALTER TABLE care_recipients ADD COLUMN IF NOT EXISTS report_summary text;"
            with engine.begin() as conn2:
                conn2.execute(text(sql_add_summary))
            print("Startup schema check: ensured care_recipients.report_summary exists.")
        except Exception as se:
            print("Startup schema check (report_summary) failed:", se)
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
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # This is needed for Authorization header
    expose_headers=["*"]  # This ensures the client can read custom headers
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
app.include_router(recipients_routes.router)

app.include_router(emotion_routes.router)

app.include_router(elderly.router)

# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Global exception handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": exc.detail or "Not authenticated"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Global exception handler for all other exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


