import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from service import MainService
from dotenv import load_dotenv

# Load Env
load_dotenv()

# Fast API:
app = FastAPI(
    title="StructCheck AI",
    description="A RAG system for PDFs using LlamaParse, LangChain, and Groq",
    version="1.0.0"
)

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

# Initialize the Main service
main_service = MainService()

# Plan Review:
@app.post('/plan-check/execute')
async def execute_plan_check(files: List[UploadFile] = File(...)):
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


        # Curate a list of potential errors
        result = await main_service.execute_plan_check(temp_file_paths)

        return {"items": result.items, "project_info": result.project_info, "success": True, }

    except Exception as e:
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

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

# Checklist Generation:
@app.post("/plan-check/structural/generate-checklist")
async def generate_checklist(file: UploadFile = File(...)):
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
            checklist_response = await main_service.generate_structural_checklist(temp_file_path)
            print(checklist_response)
            return checklist_response

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_file_path}: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing design: {str(e)}")

# Todo: Remove this route. Temporarily added until render service is hosted on free tier.
@app.get('/check-connection')
async def check_connection():
    """Check connection endpoint to verify server is running"""
    return True

@app.get("/")
def root():
    return {"status": "ok"}