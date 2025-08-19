import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Literal
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from service import MainService
from dotenv import load_dotenv
from database import connect_to_mongo, close_mongo_connection
from auth import get_current_user, get_current_user_optional, FirebaseUser


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


class UserProjectsRequest(BaseModel):
    user_id: str
    page: Optional[int] = 1
    limit: Optional[int] = 4
    search: Optional[str] = ""
    status: Optional[str] = None


class UpdateProjectStatusPayload(BaseModel):
    id: str
    status: Literal["in_progress", "completed", "under_review", "cancelled"]


class DeleteProjectRequest(BaseModel):
    id: str
    user_id: str


class DeleteQARequest(BaseModel):
    id: str
    project_id: str
    user_id: str


class DeleteChecklistRequest(BaseModel):
    checklist_id: str
    project_id: str
    user_id: str


# Load Env
load_dotenv()

main_service = None


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


# Helper function to validate user access
def validate_user_access(current_user: FirebaseUser, requested_user_id: str):
    """Validate that the current user can access the requested user's data"""
    if current_user.uid != requested_user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: You can only access your own data"
        )


# Todo: Remove this route. Temporarily added until render service is hosted on free tier.
@app.get('/check-connection')
async def check_connection():
    """Check connection endpoint to verify server is running"""
    return True


# Ingest Comments:
@app.post("/knowledgebase/structural/ingest-city-comments")
async def ingest_city_comments(
        files: List[UploadFile] = File(...),
        current_user: FirebaseUser = Depends(get_current_user)
):
    """
    Ingest city comment documents into the RAG system.
    Requires authentication.
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
async def create_project(
        request: dict,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        # Validate that user_id matches authenticated user
        validate_user_access(current_user, request.get('user_id'))

        new_project = await main_service.create_project(project_data=request)
        return new_project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@app.post('/projects/structural/get-all')
async def get_projects_by_user(
        request: UserProjectsRequest,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        # Validate that user_id matches authenticated user
        validate_user_access(current_user, request.user_id)

        result = await main_service.get_projects_by_user(
            user_id=request.user_id,
            page=request.page,
            limit=request.limit,
            search=request.search,
            status=request.status
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@app.get("/projects/structural/{id}")
async def get_project_by_id(
        id: str,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        project = await main_service.get_project_by_id(project_id=id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Validate that the project belongs to the authenticated user
        validate_user_access(current_user, project.get('user_id'))

        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@app.get("/projects/structural/{id}/qas")
async def get_project_qas(
        id: str,
        page: int = 1,
        limit: int = 4,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        # First, get the project to validate user access
        project = await main_service.get_project_by_id(project_id=id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        validate_user_access(current_user, project.get('user_id'))

        qas = await main_service.get_project_qas(project_id=id, page=page, limit=limit)
        return qas
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Project QAs: {str(e)}")


@app.get("/projects/structural/{id}/checklists")
async def get_project_checklists(
        id: str,
        page: int = 1,
        limit: int = 4,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        # First, get the project to validate user access
        project = await main_service.get_project_by_id(project_id=id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        validate_user_access(current_user, project.get('user_id'))

        checklists = await main_service.get_project_checklists(project_id=id, page=page, limit=limit)
        return checklists
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Project Checklists: {str(e)}")


@app.post("/projects/structural/update", response_model=dict)
async def update_project_status(
        payload: UpdateProjectStatusPayload,
        current_user: FirebaseUser = Depends(get_current_user)
):
    """
    Handles the API request to update a project's status.
    """
    try:
        # First, get the project to validate user access
        project = await main_service.get_project_by_id(project_id=payload.id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        validate_user_access(current_user, project.get('user_id'))

        # Call the service layer to perform the business logic
        updated_project = await main_service.update_project_status(
            project_id=payload.id,
            new_status=payload.status
        )

        return updated_project

    except Exception as e:
        # Catch any exceptions bubbled up from the service layer
        raise HTTPException(status_code=500, detail=f"Failed to update project status: {str(e)}")


@app.post("/projects/structural/delete")
async def delete_project_by_id(
        request: DeleteProjectRequest,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        validate_user_access(current_user, request.user_id)

        deleted_project = await main_service.delete_project(project_id=request.id, user_id=request.user_id)
        return deleted_project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Delete Project: {str(e)}")


# API Routes for Plan QA:
@app.post('/qas/structural/execute')
async def execute_qa(
        files: List[UploadFile] = File(...),
        user_id: str = Form(...),
        project_id: str = Form(...),
        title: str = Form(...),
        current_user: FirebaseUser = Depends(get_current_user)
):
    temp_file_paths = []

    try:
        validate_user_access(current_user, user_id)

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
async def get_qa_by_id(
        id: str,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        qa = await main_service.get_qa_by_id(qa_id=id)
        if not qa:
            raise HTTPException(status_code=404, detail="QA not found")

        # Validate that the QA belongs to the authenticated user
        validate_user_access(current_user, qa.get('user_id'))

        return qa
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get QA Run: {str(e)}")


@app.post("/qas/structural/delete")
async def delete_qa_by_id(
        request: DeleteQARequest,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        validate_user_access(current_user, request.user_id)

        deleted_project = await main_service.delete_qa(
            qa_id=request.id,
            project_id=request.project_id,
            user_id=request.user_id
        )
        return deleted_project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Delete QA: {str(e)}")


# API Routes for Users:
@app.post('/users/signin-google')
async def google_sign_in(request: dict):
    """Create a user - This endpoint doesn't require authentication as it's for sign-in"""
    try:
        created_user = await main_service.google_sign_in(user_data=request)
        return created_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating a user: {str(e)}")


# API Routes for Checklists:
@app.post("/checklists/structural/generate")
async def generate_checklist(
        file: UploadFile = File(...),
        user_id: str = Form(...),
        project_id: str = Form(...),
        title: str = Form(...),
        state: str = Form(...),
        city: str = Form(...),
        current_user: FirebaseUser = Depends(get_current_user)
):
    """
    Analyze a structural design document and generate a contextual checklist based on past city comments.
    """
    try:
        validate_user_access(current_user, user_id)

        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Generate checklist
            checklist_response = await main_service.generate_structural_checklist(temp_file_path, user_id, project_id,
                                                                                  state, city)
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
async def get_checklist_by_id(
        id: str,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        checklist = await main_service.get_checklist_by_id(checklist_id=id)
        if not checklist:
            raise HTTPException(status_code=404, detail="Checklist not found")

        # Validate that the checklist belongs to the authenticated user
        validate_user_access(current_user, checklist.get('user_id'))

        return checklist
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Checklist: {str(e)}")


@app.post("/checklists/structural/delete")
async def delete_checklist(
        request: DeleteChecklistRequest,
        current_user: FirebaseUser = Depends(get_current_user)
):
    try:
        validate_user_access(current_user, request.user_id)

        return await main_service.delete_checklist(
            checklist_id=request.checklist_id,
            project_id=request.project_id,
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Delete Checklist: {str(e)}")


@app.post("/dashboard/structural/stats")
async def get_dashboard_stats(
        request: UserIdRequest,
        current_user: FirebaseUser = Depends(get_current_user)
):
    """
    Get dashboard statistics for a user
    """
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        validate_user_access(current_user, request.user_id)

        stats = await main_service.get_dashboard_stats(user_id=request.user_id)
        return stats.dict()
    except Exception as e:
        print(f"Dashboard stats error for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard stats")


@app.get("/")
def root():
    return {"status": "ok"}