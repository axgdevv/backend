import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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