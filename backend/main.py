from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import engine
import tables.users as user_tables
import routes.users as user_routes

user_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:8080", "http://127.0.0.1:8080"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to CareTaker API", "status": "active"}

# Include routers without prefix since we're handling it in the router
app.include_router(user_routes.router)




