from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from models.users import ResponseSchema, Register, Login
from tables.users import CareTaker, CareRecipient
from config import get_db, ACCESS_TOKEN_EXPIRE_MINUTES
from repository.users import UsersRepo, JWTRepo
from utils.email import send_registration_email

router = APIRouter(tags=['Authentication'])

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

        # Add care recipients
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

        db.commit()

        # Send registration email
        await send_registration_email(request.email, request.username)

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": caretaker.username}, expires_delta=access_token_expires
        )

        return ResponseSchema(
            code=200,
            status="success",
            message="Caretaker registered successfully!",
            result={"access_token": token, "token_type": "bearer"}
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


# ---------- LOGIN ----------
@router.post('/login', response_model=ResponseSchema)
async def login(request: Login, db: Session = Depends(get_db)):
    try:
        user = UsersRepo.find_by_username(db, CareTaker, request.username)
        if not user or request.password != user.password:
            raise HTTPException(status_code=400, detail="Invalid username or password")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": user.username}, expires_delta=access_token_expires
        )

        return ResponseSchema(
            code=200,
            status="success",
            message="Login successful",
            result={"access_token": token, "token_type": "bearer"}
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
