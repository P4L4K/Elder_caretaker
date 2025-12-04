import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class FallDetector:
    def __init__(self):
        # Initialize any models or parameters here
        self.min_contour_area = 500  # Minimum contour area to consider for fall detection
        self.aspect_ratio_threshold = 0.6  # Aspect ratio threshold for fall detection
        self.movement_threshold = 30  # Minimum movement to detect
        
        # Background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.prev_frame = None
        
    def process_frame(self, frame, frame_number: int) -> Tuple[bool, float, dict]:
        """
        Process a single frame for fall detection
        Returns: (is_fall, confidence, metadata)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Initialize variables
            is_fall = False
            confidence = 0.0
            metadata = {
                'frame_number': frame_number,
                'contour_areas': [],
                'aspect_ratios': [],
                'movement_detected': False
            }
            
            # Apply background subtraction
            fgmask = self.fgbg.apply(gray)
            
            # Threshold the foreground mask
            _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
            
            # Remove noise
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                    
                metadata['contour_areas'].append(area)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                metadata['aspect_ratios'].append(aspect_ratio)
                
                # Check aspect ratio (fall typically has lower aspect ratio)
                if aspect_ratio < self.aspect_ratio_threshold:
                    confidence += 0.5
                    
                # Check area (larger area might indicate a fall)
                if area > 5000:  # Threshold for significant object
                    confidence += 0.3
                
                # Draw bounding box (for visualization)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Check for movement if we have a previous frame
                if self.prev_frame is not None:
                    # Calculate optical flow or simple frame difference
                    movement = self.detect_movement(self.prev_frame, gray, (x, y, w, h))
                    if movement > self.movement_threshold:
                        metadata['movement_detected'] = True
                        confidence += 0.2
            
            # Update previous frame
            self.prev_frame = gray.copy()
            
            # Normalize confidence to [0, 1] range
            confidence = min(1.0, confidence)
            
            # If confidence is high enough, mark as fall
            if confidence > 0.6:  # Threshold for fall detection
                is_fall = True
            
            return is_fall, confidence, metadata
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return False, 0.0, {'error': str(e)}
    
    def detect_movement(self, prev_frame: np.ndarray, current_frame: np.ndarray, bbox: tuple) -> float:
        """Detect movement between frames in the specified bounding box"""
        x, y, w, h = bbox
        prev_roi = prev_frame[y:y+h, x:x+w]
        curr_roi = current_frame[y:y+h, x:x+w]
        
        # Resize if needed for consistent size
        if prev_roi.size == 0 or curr_roi.size == 0:
            return 0.0
            
        if prev_roi.shape != curr_roi.shape:
            return 0.0
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_roi, curr_roi)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate movement metric (sum of non-zero pixels)
        movement = np.sum(thresh) / 255.0
        return movement
        
    def reset(self):
        """Reset the detector state"""
        self.prev_frame = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


def process_video_for_falls(video_path: str, output_path: str) -> dict:
    """
    Process a video file for fall detection
    Returns a dictionary with processing results
    """
    # Initialize detector
    detector = FallDetector()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'status': 'error',
            'message': f'Could not open video file: {video_path}'
        }
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_number = 0
    falls_detected = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        is_fall, confidence, metadata = detector.process_frame(frame, frame_number)
        
        # Add text overlay
        status = "NORMAL"
        color = (0, 255, 0)  # Green
        
        if is_fall:
            status = "FALL DETECTED!"
            color = (0, 0, 255)  # Red
            
            # Record fall event
            falls_detected.append({
                'frame': frame_number,
                'time': frame_number / fps,
                'confidence': confidence,
                'metadata': metadata
            })
        
        # Add status text to frame
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Write frame to output video
        out.write(frame)
        frame_number += 1
        
        # Print progress
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    return {
        'status': 'success',
        'processed_video': output_path,
        'total_frames': frame_number,
        'falls_detected': falls_detected,
        'fps': fps,
        'duration': frame_number / fps if fps > 0 else 0
    }
