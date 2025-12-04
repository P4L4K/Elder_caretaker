from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set in .env file")

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

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is not set in .env file")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Audio Pipeline Config
AUDIO_CHUNK_SIZE = 4096
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
AUDIO_FORMAT = "paInt16"
AUDIO_VAD_THRESHOLD = 800
AUDIO_MAX_SILENCE_CHUNKS = 30
AUDIO_TARGET_SR = 16000