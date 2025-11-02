from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from typing import List
from pydantic import EmailStr
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

conf = ConnectionConfig(
    MAIL_USERNAME = os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD"),
    MAIL_FROM = os.getenv("MAIL_FROM"),
    MAIL_PORT = int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER = os.getenv("MAIL_SERVER"),
    MAIL_TLS = True,
    MAIL_SSL = False,
    USE_CREDENTIALS = True
)

async def send_registration_email(email: EmailStr, username: str):
    message = MessageSchema(
        subject="Welcome to CareTaker!",
        recipients=[email],
        body=f"""
        <html>
            <body>
                <h1>Welcome to CareTaker, {username}!</h1>
                <p>Thank you for registering with CareTaker. Your account has been successfully created.</p>
                <p>You can now log in to your account and start managing care for your loved ones.</p>
                <p>Best regards,<br>The CareTaker Team</p>
            </body>
        </html>
        """,
        subtype="html"
    )

    fm = FastMail(conf)
    await fm.send_message(message)