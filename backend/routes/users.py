from fastapi import APIRouter, Depends, HTTPException, status, Header, Form
from sqlalchemy.orm import Session
from datetime import timedelta
import json
from typing import Optional, List

from models.users import ResponseSchema, Register, Login
from tables.users import CareTaker, CareRecipient
from config import get_db, ACCESS_TOKEN_EXPIRE_MINUTES
from repository.users import UsersRepo, JWTRepo
from utils.email import send_registration_email

router = APIRouter(tags=['Authentication'])


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        print("No Authorization header provided")
        return None
    
    try:
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            print(f"Invalid Authorization header format: {auth_header[:50]}...")
            return None
            
        token = parts[1]
        if not token:
            print("Empty token")
            return None
            
        print(f"Attempting to decode token: {token[:10]}...")
        decoded = JWTRepo.decode_token(token)
        
        if not decoded or not isinstance(decoded, dict):
            print("Invalid token format after decoding")
            return None
            
        username = decoded.get('sub')
        print(f"Successfully authenticated user: {username}")
        return username
        
    except Exception as e:
        print(f"Token validation error: {str(e)}")
        return None


@router.get('/profile')
async def profile(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    try:
        username = _get_username_from_auth(authorization)
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

        user = UsersRepo.find_by_username(db, CareTaker, username)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        recipients = []
        for r in user.care_recipients:
            recipients.append({
                'id': r.id,
                'full_name': r.full_name,
                'email': r.email,
                'phone_number': r.phone_number,
                'age': r.age,
                'gender': r.gender.value if r.gender else None,
                'respiratory_condition_status': bool(r.respiratory_condition_status),
                'report_summary': r.report_summary
            })

        return {
            'status': 'success',
            'caretaker': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'phone_number': user.phone_number,
                'face_registered': user.face_descriptor is not None
            },
            'care_recipients': recipients
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile: {str(e)}")

# ---------- SIGNUP ----------
@router.post('/signup', response_model=ResponseSchema)
async def signup(request: Register, db: Session = Depends(get_db)):
    try:
        # Check if caretaker already exists
        existing_user = UsersRepo.find_by_username(db, CareTaker, request.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Create CareTaker entry (plain-text password)
        caretaker = CareTaker(
            email=request.email,
            username=request.username,
            phone_number=request.phone_number,
            password=request.password,  # no hashing
            full_name=request.full_name
        )
        db.add(caretaker)
        db.commit()
        db.refresh(caretaker)

        # Add care recipients and keep references so we can return their IDs
        created_recipients = []
        for recipient in request.care_recipients:
            new_recipient = CareRecipient(
                caretaker_id=caretaker.id,
                full_name=recipient.full_name,
                email=recipient.email,
                phone_number=recipient.phone_number,
                age=recipient.age,
                gender=recipient.gender,
                respiratory_condition_status=recipient.respiratory_condition_status
            )
            db.add(new_recipient)
            created_recipients.append(new_recipient)

        db.commit()
        for r in created_recipients:
            db.refresh(r)

        # Send registration email
        await send_registration_email(request.email, request.username)

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": caretaker.username}, expires_delta=access_token_expires
        )

        # Return token and created recipient metadata (ids) so client can upload files
        recipients_out = [{
            'id': r.id,
            'full_name': r.full_name,
            'email': r.email
        } for r in created_recipients]

        return ResponseSchema(
            code=200,
            status="success",
            message="Caretaker registered successfully!",
            result={"access_token": token, "token_type": "bearer", "care_recipients": recipients_out}
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


# ---------- NEW ENDPOINT: REGISTER WITH FACE ----------
@router.post('/register-with-face')
async def register_with_face(
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    phone_number: str = Form(...),
    password: str = Form(...),
    face_descriptor: str = Form(None),
    recipient_name: List[str] = Form(...),
    recipient_email: List[str] = Form([]),
    recipient_phone: List[str] = Form([]),
    recipient_age: List[int] = Form(...),
    recipient_gender: List[str] = Form(...),
    recipient_condition: List[str] = Form(...),
    db: Session = Depends(get_db)
):
    """
    New endpoint for registration with face data from webcam
    """
    try:
        # Check if caretaker already exists
        existing_user = UsersRepo.find_by_username(db, CareTaker, username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Parse face descriptor if provided
        face_descriptor_data = None
        if face_descriptor and face_descriptor != 'null':
            try:
                face_descriptor_data = json.loads(face_descriptor)
                print(f"Face descriptor received with {len(face_descriptor_data)} elements")
            except json.JSONDecodeError:
                print("Warning: Invalid face descriptor format - continuing without face data")

        # Create CareTaker entry with face descriptor
        caretaker = CareTaker(
            email=email,
            username=username,
            phone_number=phone_number,
            password=password,
            full_name=full_name,
            face_descriptor=face_descriptor_data  # Store the face data
        )
        db.add(caretaker)
        db.commit()
        db.refresh(caretaker)

        # Add care recipients
        created_recipients = []
        for i in range(len(recipient_name)):
            new_recipient = CareRecipient(
                caretaker_id=caretaker.id,
                full_name=recipient_name[i],
                email=recipient_email[i] if i < len(recipient_email) else None,
                phone_number=recipient_phone[i] if i < len(recipient_phone) else None,
                age=recipient_age[i],
                gender=recipient_gender[i],
                respiratory_condition_status=recipient_condition[i].lower() == 'true'
            )
            db.add(new_recipient)
            created_recipients.append(new_recipient)

        db.commit()
        for r in created_recipients:
            db.refresh(r)

        # Send registration email
        await send_registration_email(email, username)

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": caretaker.username}, expires_delta=access_token_expires
        )

        # Return response with recipients
        recipients_out = [{
            'id': r.id,
            'full_name': r.full_name,
            'email': r.email
        } for r in created_recipients]

        return {
            "code": 200,
            "status": "success",
            "message": "Caretaker registered successfully!" + (" Face data stored." if face_descriptor_data else ""),
            "result": {
                "access_token": token, 
                "token_type": "bearer",
                "face_registered": bool(face_descriptor_data),
                "care_recipients": recipients_out
            }
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


# ---------- UPDATE FACE FOR EXISTING USER ----------
@router.post('/update-face')
async def update_face(
    face_descriptor: str = Form(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Update face descriptor for an existing user
    """
    try:
        username = _get_username_from_auth(authorization)
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

        user = UsersRepo.find_by_username(db, CareTaker, username)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        # Parse face descriptor
        try:
            face_descriptor_data = json.loads(face_descriptor)
            user.face_descriptor = face_descriptor_data
            db.commit()
            
            return {
                "code": 200,
                "status": "success", 
                "message": "Face descriptor updated successfully",
                "result": {"face_registered": True}
            }
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid face descriptor format")

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update face: {str(e)}")


# ---------- LOGIN ----------
@router.post('/login', response_model=ResponseSchema)
async def login(request: Login, db: Session = Depends(get_db)):
    try:
        print(f"Login attempt for username: {request.username}")
        
        # Check if user exists
        user = UsersRepo.find_by_username(db, CareTaker, request.username)
        if not user:
            print(f"Login failed: User '{request.username}' not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )

        # Verify password (plain text comparison for now)
        if user.password != request.password:
            print(f"Login failed: Invalid password for user '{request.username}'")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": user.username},
            expires_delta=access_token_expires
        )
        
        print(f"Login successful for user: {user.username}")
        print(f"Generated token: {token[:10]}...")

        # Return user details along with the token
        return ResponseSchema(
            code=200,
            status="success",
            message="Login successful",
            result={
                "access_token": token, 
                "token_type": "bearer",
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "face_registered": user.face_descriptor is not None
                }
            }
        )

    except HTTPException as e:
        print(f"HTTPException during login: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Login failed: {str(e)}"
        print(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=error_msg
        )