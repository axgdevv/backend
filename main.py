import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime

import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from service import MainService
from dotenv import load_dotenv
from database import connect_to_mongo, close_mongo_connection


class ChecklistRequest(BaseModel):
    checklist_id: str
    user_id: str
    checklist_data: Dict


class UserRequest(BaseModel):
    firebase_uid: str
    email: str
    full_name: str
    email_verified: bool
    last_sign_in_at: str
    photo_url: str
    created_at: str

class UserIdRequest(BaseModel):
    user_id: str

class ChecklistIdRequest(BaseModel):
    checklist_id: str

class PlanCheckIdRequest(BaseModel):
    plan_check_id: str

class ProjectInfo(BaseModel):
    project_name: str
    client_name: str

class UpdatePlanCheckRequest(BaseModel):
    plan_check_id: str
    project_info: ProjectInfo

class UpdateChecklistRequest(BaseModel):
    checklist_id: str
    project_info: ProjectInfo

# Load Env
load_dotenv()

main_service=None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_service

    # Connect to MongoDB
    await connect_to_mongo()

    # Initialize main service
    main_service = MainService()

    yield  # app is running

    # Shutdown: close MongoDB connection
    await close_mongo_connection()
app = FastAPI(lifespan=lifespan)

# CORS
client_url = os.getenv("CLIENT_URL", "http://localhost:3000")
origins = [client_url]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ingest Comments:
@app.post("/plan-check/structural/ingest-city-comments")
async def ingest_city_comments(files: List[UploadFile] = File(...)):
    """
    Ingest city comment documents into the RAG system.

    Args:
        files: List of PDF files containing city comments

    Returns:
        Status of ingestion process
    """
    temp_file_paths = []
    try:
        # Save uploaded files temporarily
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_paths.append(temp_file.name)

        # Ingest comments into RAG system
        result = await main_service.ingest_comments(temp_file_paths)

        # Clean up temporary files
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_path}: {str(e)}")

        return JSONResponse(content=result)

    except Exception as e:
        # Clean up temporary files in case of error
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

# Todo: Remove this route. Temporarily added until render service is hosted on free tier.
@app.get('/check-connection')
async def check_connection():
    """Check connection endpoint to verify server is running"""
    return True


#################### CRUD for Plan Review ####################

@app.post('/plan-check/execute')
async def execute_plan_check(files: List[UploadFile] = File(...), user_id: str = Form(...), plan_check_id: str = Form(...)):
    temp_file_paths = []

    print(user_id, plan_check_id)
    try:

        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_paths.append(temp_file.name)


        # Curate a list of potential errors
        result = await main_service.execute_plan_check(temp_file_paths, user_id, plan_check_id)

        return {"items": result.items, "project_info": result.project_info, "success": True, }

    except Exception as e:
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post('/plan-check/structural/get-user-plan-checks')
async def get_checklists_by_user(request: UserIdRequest):
    try:
        plan_checks = await main_service.get_plan_checks_by_user(user_id=request.user_id)
        return plan_checks
    except Exception as e:
        raise (HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}"))

@app.post('/plan-check/structural/get-plan-check-by-id')
async def get_checklist_by_id(request: PlanCheckIdRequest):
    try:
        plan_check = await main_service.get_plan_check_by_id(plan_check_id=request.plan_check_id)
        return plan_check
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}")

@app.post('/plan-check/structural/update-plan-check')
async def update_checklist(request: UpdatePlanCheckRequest):
    try:
        plan_check = await main_service.update_plan_check(plan_check_id=request.plan_check_id, project_info=request.project_info)
        return plan_check
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}")

@app.post('/plan-check/structural/get-user-plan-checks')
async def get_checklists_by_user(request: UserIdRequest):
    try:
        plan_checks = await main_service.get_plan_checks_by_user(user_id=request.user_id)
        return plan_checks
    except Exception as e:
        raise (HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}"))


#################### CRUD for Users ####################
@app.post('/users/signin-google')
async def google_sign_in(request: dict):
    """Create a checklist"""
    try:

        created_user = await main_service.google_sign_in(user_data=request)
        return created_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating a checklist: {str(e)}")


#################### CRUD for Checklist ####################
@app.post("/plan-check/structural/generate-checklist")
async def generate_checklist(file: UploadFile = File(...), user_id: str = Form(...), checklist_id: str = Form(...)):
    """
    Analyze a structural design document and generate a contextual checklist based on past city comments.

    Args:
        file: PDF file containing the structural design

    Returns:
        Generated checklist specific to the design
    """

    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Generate checklist
            checklist_response = await main_service.generate_structural_checklist(temp_file_path, user_id, checklist_id)
            return checklist_response

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_file_path}: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing design: {str(e)}")

@app.post('/plan-check/structural/get-user-checklists')
async def get_checklists_by_user(request: UserIdRequest):
    try:
        checklists = await main_service.get_checklists_by_user(user_id=request.user_id)
        return checklists
    except Exception as e:
        raise (HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}"))

@app.post('/plan-check/structural/get-checklist-by-id')
async def get_checklist_by_id(request: ChecklistIdRequest):
    try:
        checklists = await main_service.get_checklist_by_id(checklist_id=request.checklist_id)
        return checklists
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}")

@app.post('/plan-check/structural/update-checklist')
async def update_checklist(request: UpdateChecklistRequest):
    try:
        checklists = await main_service.update_checklist(checklist_id=request.checklist_id, project_info=request.project_info)
        return checklists
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}")

@app.post("/plan-check/structural/delete-checklist")
async def delete_checklist_route(
    request: ChecklistIdRequest,
):
    return await main_service.delete_checklist(request.checklist_id)

@app.get("/")
def root():
    return {"status": "ok"}