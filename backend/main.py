from fastapi import FastAPI, Request, status, UploadFile, File, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import subprocess
import os
import uuid
import shutil
import time
import re
from typing import Dict, Optional, List
import json
from datetime import datetime, timedelta
import psutil
from pathlib import Path
import base64
from fastapi.responses import FileResponse
# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import database models and tables
import tables.users as user_tables
import tables.recordings as recordings_tables
import tables.medical_reports as med_reports_tables

# Import routes
import routes.users as user_routes
import routes.recordings as recordings_routes
import routes.recipients as recipients_routes
import routes.emergency as emergency_routes
from routes import elderly

# Create database tables
user_tables.Base.metadata.create_all(bind=engine)
recordings_tables.Base.metadata.create_all(bind=engine)
med_reports_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount the model directory to be served at /api/model
app.mount("/api/model", StaticFiles(directory="../model"), name="model")

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
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sdk', 'model'))
if os.path.exists(model_dir):
    app.mount("/model", StaticFiles(directory=model_dir), name="model")

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

# Include routers with proper prefixes
app.include_router(user_routes.router, prefix="/api")
app.include_router(recordings_routes.router, prefix="/api")
app.include_router(recipients_routes.router, prefix="/api")
app.include_router(emergency_routes.router, prefix="/api")
app.include_router(elderly.router, prefix="/api")

# Debug: Print all registered routes
print("\n=== Registered Routes ===")
for route in app.routes:
    if hasattr(route, "path") and hasattr(route, "methods"):
        methods = ", ".join(route.methods)
        print(f"{route.path} - {methods}")
print("=======================\n")

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


# Add this dictionary to track processes
process_status = {}

# Add this endpoint
@app.get("/api/fall-detection/status/{process_id}")
async def get_status(process_id: int):
    if process_id not in process_status:
        raise HTTPException(status_code=404, detail="Process not found")
    
    status = process_status[process_id]
    
    # The error suggests there's a reference to 'time' here that's not defined
    # It should be using time.time() or similar
    if "start_time" in status:
        elapsed = time.time() - status["start_time"]
        status["elapsed"] = round(elapsed, 2)
    
    return status

# Add this with your other routes
@app.post("/api/fall-detection/process-video")
async def process_video( file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    process_id = None
    try:
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save the uploaded file
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the video using your fall detection script
        output_dir = os.path.join(os.path.dirname(__file__), "output_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"output_{filename}"
        output_path = os.path.join(output_dir, output_file)
        cmd = f"python fall_detection.py --video {file_path} --show --output {output_path}"
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Store process info
        process_id = process.pid
        process_status[process_id] = {
            "status": "processing",
            "output_file": output_file,
            "output_path": output_path,
            "progress": 0,
            "error": None
        }
        
        # Add cleanup task
        if background_tasks:
            background_tasks.add_task(
                monitor_process,
                process_id=process_id,
                process=process,
                output_file=output_file
            )
        
        return {
            "process_id": process_id,
            "status": "processing",
            "output_file": output_file
        }
        
    except Exception as e:
       if process_id in process_status:
            process_status[process_id]["status"] = "error"
            process_status[process_id]["error"] = str(e)
       raise HTTPException(status_code=500, detail=str(e))

# Add this helper function
async def monitor_process(process_id: int, process: subprocess.Popen, output_file: str):
    """
    Monitor the fall detection process and update the status.
    This runs in the background and updates the process status.
    """
    try:
        # Wait for the process to complete
        stdout_data = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stdout_data.append(output.strip())
                print(output.strip())  # Log output
                
                # Parse progress if available
                if "Progress:" in output:
                    try:
                        progress = int(output.split("Progress:")[1].strip().strip("%"))
                        if process_id in process_status:
                            process_status[process_id]["progress"] = progress
                    except (IndexError, ValueError):
                        pass
        
        # Process completed, update status
        if process_id in process_status:
            if process.returncode == 0:
                process_status[process_id].update({
                    "status": "completed",
                    "progress": 100,
                    "output": "\n".join(stdout_data)
                })
                
                # Parse fall detection results if available
                falls_detected = []
                for i, line in enumerate(stdout_data):
                    if "FALL DETECTED!" in line:
                        try:
                            # Get the time from the next line
                            time_line = stdout_data[i + 1]
                            time_str = time_line.split("Video Time:")[1].strip()
                            
                            # Get angle from the line after next, handling non-ASCII characters
                            if i + 2 < len(stdout_data) and "Angle:" in stdout_data[i + 2]:
                                angle_line = stdout_data[i + 2]
                                # Clean the angle string by removing non-ASCII characters
                                angle_str = angle_line.split("Angle:")[1].split("°")[0].strip()
                                angle = float(angle_str.replace('Â', '').strip())  # Remove non-ASCII characters
                                
                                # Use angle as a proxy for confidence (normalized to 0-1)
                                confidence = min(1.0, angle / 45.0)
                                
                                # Convert time string to seconds
                                h, m, s = map(float, time_str.split(":"))
                                timestamp_seconds = h * 3600 + m * 60 + s
                                
                                falls_detected.append({
                                    "timestamp": time_str,
                                    "timestamp_seconds": timestamp_seconds,
                                    "confidence": confidence,
                                    "angle": angle
                                })
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing fall detection output: {e}")
                
                if falls_detected:
                    process_status[process_id]["falls_detected"] = falls_detected
                    process_status[process_id]["has_falls"] = True
                else:
                    process_status[process_id]["has_falls"] = False
                    
            else:
                process_status[process_id].update({
                    "status": "error",
                    "error": f"Process failed with return code {process.returncode}",
                    "output": "\n".join(stdout_data)
                })
    except Exception as e:
        if process_id in process_status:
            process_status[process_id].update({
                "status": "error",
                "error": str(e)
            })
        print(f"Error in monitor_process: {e}")
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(f"temp_uploads/{output_file}"):
                os.remove(f"temp_uploads/{output_file}")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "output_videos")
os.makedirs(output_dir, exist_ok=True)

# Serve output videos from the output directory
app.mount("/videos", StaticFiles(directory=output_dir), name="videos")

@app.get("/videos/{filename}")
async def get_video(filename: str, request: Request):
    video_path = os.path.join("output_videos", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_size = os.path.getsize(video_path)
    start = 0
    end = file_size - 1
    
    range_header = request.headers.get('range')
    if range_header:
        bytes_range = range_header.replace('bytes=', '').split('-')
        start = int(bytes_range[0]) if bytes_range[0] else 0
        if len(bytes_range) > 1 and bytes_range[1]:
            end = int(bytes_range[1])
    
    chunk_size = end - start + 1
    with open(video_path, 'rb') as f:
        f.seek(start)
        data = f.read(chunk_size)
    
    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(chunk_size),
        'Content-Type': 'video/mp4',
    }
    
    return Response(content=data, status_code=206, headers=headers, media_type="video/mp4")
    video_path = os.path.join("output_videos", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get file size for range requests
    file_size = os.path.getsize(video_path)
    start = 0
    end = file_size - 1
    
    # Handle range requests
    range_header = request.headers.get('range')
    if range_header:
        bytes_range = range_header.replace('bytes=', '').split('-')
        start = int(bytes_range[0]) if bytes_range[0] else 0
        if len(bytes_range) > 1 and bytes_range[1]:
            end = int(bytes_range[1])
    
    # Read the requested chunk of the file
    chunk_size = end - start + 1
    with open(video_path, 'rb') as f:
        f.seek(start)
        data = f.read(chunk_size)
    
    # Set appropriate headers
    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(chunk_size),
        'Content-Type': 'video/mp4',
    }
    
    return Response(content=data, status_code=206, headers=headers, media_type="video/mp4")

# Add these imports at the top of the file with other imports
from fastapi import APIRouter, HTTPException
from datetime import datetime
from weather import WeatherPredictionModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Weather router
weather_router = APIRouter()
weather_model = None

@weather_router.get("/weather/current")
async def get_current_weather():
    global weather_model
    
    if not weather_model:
        return {
            "error": "Weather service not available",
            "status": "error"
        }
    
    try:
        data = weather_model.fetch_data()
        if not data:
            return {
                "error": "Failed to fetch weather data. Please try again later.",
                "status": "error"
            }
            
        current = data.get('current')
        if not current:
            return {
                "error": "Invalid weather data received",
                "status": "error"
            }
            
        return {
            "temperature": current.get('temp_c', 'N/A'),
            "humidity": current.get('humidity', 'N/A'),
            "aqi": current.get('air_quality', {}).get('us-epa-index', 0),
            "condition": current.get('condition', {}).get('text', 'Unknown'),
            "location": weather_model.city,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this with your other router includes (usually where you have other app.include_router() calls)
app.include_router(weather_router, prefix="/api")

# Add this with your other startup event handlers
@app.on_event("startup")
async def startup_event():
    global weather_model
    try:
        API_KEY = os.getenv("WEATHER_API_KEY", "628d4985109c4f6baa3182527250312")
        DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Jammu")
        weather_model = WeatherPredictionModel(API_KEY, DEFAULT_CITY)
        print("✅ Weather service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize weather service: {e}")

# Make sure this is at the end of the file
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)