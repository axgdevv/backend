import os
import json
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth
from pydantic import BaseModel


class FirebaseUser(BaseModel):
    uid: str
    email: str
    email_verified: bool
    name: Optional[str] = None
    picture: Optional[str] = None


class FirebaseAuth:
    def __init__(self):
        self.security = HTTPBearer()
        self._initialize_firebase()

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        if not firebase_admin._apps:
            # Load Firebase credentials from environment variables
            firebase_config = {
                "type": os.getenv("FIREBASE_TYPE"),
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n') if os.getenv("FIREBASE_PRIVATE_KEY") else None,
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
                "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
                "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
                "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN", "googleapis.com")
            }

            # Remove None values
            firebase_config = {k: v for k, v in firebase_config.items() if v is not None}

            if not firebase_config.get("project_id"):
                raise ValueError("Firebase configuration is incomplete. Please check your environment variables.")

            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)

    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> FirebaseUser:
        """Verify Firebase ID token and return user info"""
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(credentials.credentials)

            # Extract user information
            firebase_user = FirebaseUser(
                uid=decoded_token['uid'],
                email=decoded_token.get('email', ''),
                email_verified=decoded_token.get('email_verified', False),
                name=decoded_token.get('name'),
                picture=decoded_token.get('picture')
            )

            return firebase_user

        except auth.InvalidIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        except auth.ExpiredIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token has expired"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}"
            )


# Global Firebase auth instance
firebase_auth = FirebaseAuth()


# Dependency function to get current user
async def get_current_user(user: FirebaseUser = Depends(firebase_auth.verify_token)) -> FirebaseUser:
    """Dependency to get current authenticated user"""
    return user


# Optional dependency that allows unauthenticated access (for public endpoints)
async def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[
    FirebaseUser]:
    """Optional dependency that allows unauthenticated access"""
    if credentials is None:
        return None

    try:
        return await firebase_auth.verify_token(credentials)
    except HTTPException:
        return None