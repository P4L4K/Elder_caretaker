from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Depends, status
from fastapi.responses import FileResponse, JSONResponse
import os
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import queue
import threading
import time
import json
from datetime import datetime

# Import our fall detection model
from ..models.fall_detection import process_video_for_falls

router = APIRouter(prefix="/api/fall-detection", tags=["Fall Detection"])

# Ensure directories exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DIR = Path("processed_videos")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# In-memory storage for processing status
processing_status = {}

# Helper function to verify JWT token
def verify_token(authorization: str = Header(...)):
    # In a real app, verify the JWT token here
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token"
        )
    return authorization.split(" ")[1]

def get_processing_status(process_id: str):
    """Get the status of a processing job"""
    return processing_status.get(process_id, {"status": "not_found", "message": "Process ID not found"})

@router.post("/process-video", response_model=Dict[str, Any])
async def process_fall_detection(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """
    Process a video file for fall detection.
    Returns a process ID that can be used to check status and get results.
    """
    # Generate a unique process ID
    process_id = str(uuid.uuid4())
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only video files (mp4, mov, avi, mkv) are supported"
        )
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = UPLOAD_DIR / filename
        output_filename = f"processed_{filename}"
        output_path = PROCESSED_DIR / output_filename
        
        # Save the uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update processing status
        processing_status[process_id] = {
            "status": "processing",
            "message": "Video upload successful, starting processing...",
            "progress": 0,
            "start_time": datetime.now().isoformat(),
            "input_file": str(input_path),
            "output_file": str(output_path)
        }
        
        # Start processing in a background thread
        def process_video():
            try:
                # Process the video
                result = process_video_for_falls(
                    video_path=str(input_path),
                    output_path=str(output_path)
                )
                
                # Update status with results
                processing_status[process_id].update({
                    "status": "completed",
                    "message": "Video processing completed",
                    "progress": 100,
                    "end_time": datetime.now().isoformat(),
                    "result": result
                })
                
            except Exception as e:
                processing_status[process_id].update({
                    "status": "error",
                    "message": f"Error processing video: {str(e)}",
                    "end_time": datetime.now().isoformat(),
                    "error": str(e)
                })
        
        # Start the processing in a separate thread
        import threading
        thread = threading.Thread(target=process_video)
        thread.start()
        
        return {
            "status": "processing",
            "process_id": process_id,
            "message": "Video upload successful. Processing has started.",
            "check_status_url": f"/api/fall-detection/status/{process_id}"
        }
        
    except Exception as e:
        # Clean up if there was an error
        if 'input_path' in locals() and input_path.exists():
            input_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing video: {str(e)}"
        )

@router.get("/status/{process_id}")
async def get_processing_status(process_id: str):
    """Get the status of a processing job"""
    status_info = processing_status.get(process_id)
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No processing job found with ID: {process_id}"
        )
    
    # If processing is complete, include the result URL
    if status_info.get("status") == "completed":
        result = status_info.get("result", {})
        if result.get("status") == "success":
            status_info["result_url"] = f"/api/fall-detection/results/{process_id}"
    
    return status_info

@router.get("/results/{process_id}")
async def get_processing_results(process_id: str):
    """Get the results of a completed processing job"""
    status_info = processing_status.get(process_id)
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No processing job found with ID: {process_id}"
        )
    
    if status_info.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Processing not yet complete. Current status: {status_info.get('status')}"
        )
    
    result = status_info.get("result", {})
    if result.get("status") != "success":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {result.get('message', 'Unknown error')}"
        )
    
    # Return the processing results
    return {
        "status": "success",
        "process_id": process_id,
        "processed_video": f"/api/fall-detection/video/{process_id}",
        "falls_detected": result.get("falls_detected", []),
        "total_frames": result.get("total_frames", 0),
        "fps": result.get("fps", 0),
        "duration": result.get("duration", 0)
    }

@router.get("/video/{process_id}")
async def get_processed_video(process_id: str):
    """Stream the processed video"""
    status_info = processing_status.get(process_id)
    if not status_info or status_info.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or processing not complete"
        )
    
    output_path = Path(status_info.get("output_file", ""))
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video file not found"
        )
    
    return FileResponse(
        str(output_path),
        media_type="video/mp4",
        filename=f"processed_{process_id}.mp4"
    )
