import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
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

class ProjectChecklistsRequest(BaseModel):
    user_id: str
    project_id: str

class ProjectRequest(BaseModel):
    project_name: str
    client_name: str
    project_type: str
    state: str
    city: str
    user_id: str
    # checklist_count: int
    # qa_count: int
    # created_at: str
    # status: str
    # domain: str

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





# Todo: Remove this route. Temporarily added until render service is hosted on free tier.
@app.get('/check-connection')
async def check_connection():
    """Check connection endpoint to verify server is running"""
    return True





# Ingest Comments:
@app.post("/knowledgebase/structural/ingest-city-comments")
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




# API Routes for Projects:
@app.post('/projects/structural/create')
async def create_project(request: dict):
    try:
        new_project = await main_service.create_project(project_data=request)
        return new_project
    except Exception as e:
        raise (HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}"))

@app.post('/projects/structural/get-all')
async def get_projects_by_user(request: UserIdRequest):
    try:
        projects = await main_service.get_projects_by_user(user_id=request.user_id)
        return projects
    except Exception as e:
        raise (HTTPException(status_code=500, detail=f"Failed to get checklists: {str(e)}"))

@app.get("/projects/structural/{id}")
async def get_project_by_id(id: str):
    try:
        project = await main_service.get_project_by_id(project_id=id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")

@app.get("/projects/structural/{id}/qas")
async def get_project_qas(id: str):
    try:
        qas = await main_service.get_project_qas(project_id=id)
        return qas
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Project QAs: {str(e)}")

@app.get("/projects/structural/{id}/checklists")
async def get_project_checklists(id: str):
    try:
        checklists = await main_service.get_project_checklists(project_id=id)
        return checklists
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Project Checklists: {str(e)}")

# Todo: Add Delete Project Route





# API Routes for Plan QA:
@app.post('/qas/structural/execute')
async def execute_qa(files: List[UploadFile] = File(...), user_id: str = Form(...), project_id: str = Form(...), title: str = Form(...)):
    temp_file_paths = []

    try:

        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_paths.append(temp_file.name)

        # Execute a QA on plan set:
        result = await main_service.execute_qa(temp_file_paths, user_id, project_id, title)

        return result

    except Exception as e:
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.get("/qas/structural/{id}")
async def get_qa_by_id(id: str):
    try:
        qa = await main_service.get_qa_by_id(qa_id=id)
        return qa
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get QA Run: {str(e)}")

# Todo: Add Delete QA Run Route
# Todo: Add Update QA Run Route





# API Routes for Users:
@app.post('/users/signin-google')
async def google_sign_in(request: dict):
    """Create a checklist"""
    try:

        created_user = await main_service.google_sign_in(user_data=request)
        return created_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating a checklist: {str(e)}")





# API Routes for Checklists:
@app.post("/checklists/structural/generate")
async def generate_checklist(file: UploadFile = File(...), user_id: str = Form(...), project_id: str = Form(...), title: str = Form(...)):
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
            checklist_response = await main_service.generate_structural_checklist(temp_file_path, user_id, project_id, title)
            return checklist_response

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_file_path}: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing design: {str(e)}")

@app.get("/checklists/structural/{id}")
async def get_checklist_by_id(id: str):
    try:
        checklist = await main_service.get_checklist_by_id(checklist_id=id)
        return checklist
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Checklist: {str(e)}")

@app.post("/checklists/structural/delete")
async def delete_checklist(
    request: ChecklistIdRequest,
):
    return await main_service.delete_checklist(request.checklist_id)

# Todo: Add Update Checklist Route




@app.post("/dashboard/structural/stats")
async def get_dashboard_stats(request: UserIdRequest):
    """
    Get dashboard statistics for a user

    Returns:
        - Total projects created this month
        - Active projects this month
        - Completed projects (all time)
        - Projects by location breakdown
        - Top issue categories from QA runs
    """
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        stats = await main_service.get_dashboard_stats(user_id=request.user_id)
        return stats.dict()
    except Exception as e:
        print(f"Dashboard stats error for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard stats")





@app.get("/")
def root():
    return {"status": "ok"}