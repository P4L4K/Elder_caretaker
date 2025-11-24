from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL ="postgresql://postgres:word123@localhost:5432/caretaker"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print("DB session failed:", e)
        raise  # re-raise so FastAPI knows
    finally:
        db.close()


#jwt
SECRET_KEY = "crie"
ALGORITHM ='HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Audio Pipeline Config
AUDIO_CHUNK_SIZE = 4096
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
AUDIO_FORMAT = "paInt16"  # Will map to pyaudio constant in capture.py
AUDIO_VAD_THRESHOLD = 800
AUDIO_MAX_SILENCE_CHUNKS = 30
AUDIO_TARGET_SR = 16000




