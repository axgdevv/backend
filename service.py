# Core libraries
import datetime
import json
import os
from typing import List, Dict, Optional
import pathlib
import gc
from contextlib import contextmanager

from bson import ObjectId
from datetime import datetime
# FastAPI:
from fastapi import HTTPException

# Pydantic
from pydantic import BaseModel
from email.utils import parsedate_to_datetime

# LangChain:
from langchain.schema import Document

# Qdrant:
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, \
    PayloadSchemaType
import uuid

# Gemini:
from google import genai
from google.genai import types

# Load Environment variable
from dotenv import load_dotenv

from database import get_database
from pymongo import ReturnDocument

load_dotenv()

# Load Gemini Client:
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))


# Plan Review:
class ProjectInfo(BaseModel):
    project_name: str
    client_name: str


class PlanCheckItem(BaseModel):
    category: str
    item: str
    description: str
    priority: str
    reference: str
    confidence: str


class PlanCheckResponse(BaseModel):
    project_info: ProjectInfo
    items: List[PlanCheckItem]


# Checklist Generation:
class ChecklistItem(BaseModel):
    category: str
    item: str
    description: str
    priority: str  # "High", "Medium", "Low"


class ChecklistResponse(BaseModel):
    project_info: ProjectInfo
    checklist_items: List[ChecklistItem]
    relevant_comments_count: int
    summary_of_key_concerns: str
    suggested_next_steps: List[str]


class DesignAnalysis(BaseModel):
    filename: str
    city: str
    project_type: str
    design_description: str
    key_elements: List[str]


# City Comments:
class Comment(BaseModel):
    text: str


class ProjectCommentsResponse(BaseModel):
    filename: str
    city: str
    project_type: str
    comments: List[Comment]


def cleanup_memory():
    """Force garbage collection to free up memory"""
    collected = gc.collect()
    print(f"Garbage collector: collected {collected} objects")


class PlanCheck:
    def __init__(self):
        self.gemini_client = client
        self.qdrant_client = None
        self.collection_name = "city_comments"

        # Don't initialize embeddings until needed - MEMORY OPTIMIZATION
        self.embeddings = None
        # Use lighter embedding model - MEMORY OPTIMIZATION
        self._embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Initialize Qdrant client (lightweight)
        self._initialize_qdrant()

    def _get_embeddings(self):
        """Lazy load embeddings only when needed - MEMORY OPTIMIZATION"""
        if self.embeddings is None:
            print("Loading embedding model (this may take a moment)...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name,
                model_kwargs={'device': 'cpu'}
            )
            print(f"Embedding model '{self._embedding_model_name}' loaded successfully")
        return self.embeddings

    def _initialize_qdrant(self):
        """Initialize Qdrant cloud client - NO TEST OPERATIONS"""
        try:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if not qdrant_url or not qdrant_api_key:
                raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables are required")

            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30,
            )

            print("Qdrant client initialized (connection will be verified on first use)")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Qdrant connection: {str(e)}")
            raise RuntimeError(f"Cannot start service without Qdrant connection: {str(e)}")

    def create_or_update_vectorstore(self, chunks: List[Document], append_mode: bool = True):
        """Create or append to vector store - loads embeddings on demand"""
        embeddings = self._get_embeddings()  # Load embeddings only when needed

        if append_mode:
            print(f"Appending {len(chunks)} new documents to existing vector store...")
        else:
            print(f"Creating new vector store with {len(chunks)} documents...")
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                print(f"Deleted existing collection '{self.collection_name}'")
            except:
                pass  # Collection might not exist

        # Check if collection exists, create if needed
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' exists with {collection_info.vectors_count} vectors")
        except Exception:
            # Collection doesn't exist, create it
            print(f"Creating collection '{self.collection_name}'...")
            sample_embedding = embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="city",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="project_type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="file_name",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"Created collection '{self.collection_name}' with dimension {embedding_dim} and payload indexes")
            except Exception as e:
                print(f"Note: Some indexes may already exist: {e}")

        try:
            points = []
            for i, chunk in enumerate(chunks):
                embedding = embeddings.embed_query(chunk.page_content)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk.page_content,
                        "city": chunk.metadata.get("city", "Unknown"),
                        "project_type": chunk.metadata.get("project_type", "Unknown"),
                        "file_name": chunk.metadata.get("file_name", "Unknown"),
                        "page": chunk.metadata.get("page", None)
                    }
                )
                points.append(point)

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            collection_info = self.qdrant_client.get_collection(self.collection_name)
            action = "appended to" if append_mode else "created in"
            print(f"Successfully {action} vector store. Total vectors: {collection_info.vectors_count}")

        except Exception as e:
            print(f"Error creating/updating vector store: {str(e)}")
            raise

    def load_vectorstore(self):
        """Check if vector store exists - lightweight operation"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Found existing Qdrant collection: {collection_info.vectors_count} vectors")
            return True
        except Exception as e:
            print(f"No existing vectorstore found - will create when needed")
            return False

    def search_similar_documents(self, query: str, limit: int = 20, city_filter: str = None,
                                 project_type_filter: str = None) -> List[Document]:
        """Search for similar documents - loads embeddings on demand"""
        embeddings = self._get_embeddings()  # Load embeddings only when needed

        try:
            query_embedding = embeddings.embed_query(query)

            filter_conditions = []
            if city_filter:
                filter_conditions.append(
                    FieldCondition(key="city", match=MatchValue(value=city_filter))
                )
            if project_type_filter:
                filter_conditions.append(
                    FieldCondition(key="project_type", match=MatchValue(value=project_type_filter))
                )

            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )

            # If no filtered results, try with less restrictive filters
            if len(search_results) == 0 and city_filter and project_type_filter:
                print("No results with both filters, trying project_type only...")
                project_only_filter = Filter(must=[
                    FieldCondition(key="project_type", match=MatchValue(value=project_type_filter))
                ])
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=project_only_filter,
                    limit=limit,
                    with_payload=True
                )

            if len(search_results) == 0:
                print("No filtered results found, searching without filters...")
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=min(limit, 10),  # Limit unfiltered results
                    with_payload=True
                )

            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload["content"],
                    metadata={
                        "city": result.payload.get("city", "Unknown"),
                        "project_type": result.payload.get("project_type", "Unknown"),
                        "file_name": result.payload.get("file_name", "Unknown"),
                        "page": result.payload.get("page", None),
                        "score": result.score
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

    def setup_retriever(self):
        """Lightweight setup - just check collection exists"""
        print("Setting up retriever...")
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Qdrant collection '{self.collection_name}' ready with {collection_info.vectors_count} vectors")
            return True
        except Exception as e:
            print(f"Collection '{self.collection_name}' doesn't exist yet - will be created when needed")
            return False

    async def process_comments(self, city_comments: List[ProjectCommentsResponse], append_mode: bool = True):
        """Process comments and create vectorstore"""
        all_documents = []
        for city_comment in city_comments:
            try:
                for comment in city_comment.comments:
                    doc = Document(
                        page_content=comment.text,
                        metadata={
                            "city": city_comment.city,
                            "project_type": city_comment.project_type,
                            "file_name": city_comment.filename,
                            "page": getattr(comment, 'page_number', None)
                        }
                    )
                    all_documents.append(doc)
            except Exception as e:
                print(f"Error processing comment: {str(e)}")
                continue

        if all_documents:
            self.create_or_update_vectorstore(all_documents, append_mode)
            setup_success = self.setup_retriever()
            if setup_success:
                action = "appended to" if append_mode else "created"
                print(f"Vector store {action} successfully!")
            else:
                print("Warning: Vector store updated but retriever setup failed")
        else:
            print("No documents to process")

    async def extract_comments_with_gemini(self, temp_file_paths: List[str]) -> list[ProjectCommentsResponse]:
        """Extract structured city comments from PDF using Gemini AI"""
        try:
            multiple_city_comments = []

            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "extract_city_comments.txt")
            with open(prompt_path, "r") as f:
                prompt = f.read()

            for temp_path in temp_file_paths:
                try:
                    filepath = pathlib.Path(temp_path)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            types.Part.from_bytes(
                                data=filepath.read_bytes(),
                                mime_type='application/pdf',
                            ),
                            prompt],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": ProjectCommentsResponse,
                        },
                    )
                    raw_json = response.candidates[0].content.parts[0].text
                    parsed_dict = json.loads(raw_json)
                    validated = ProjectCommentsResponse(**parsed_dict)
                    multiple_city_comments.append(validated)
                except Exception as e:
                    print(f"Error processing file {temp_path}: {str(e)}")
            return multiple_city_comments
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting comments from documents: {str(e)}")

    async def analyze_structural_design_with_gemini(self, design_file_path: str) -> DesignAnalysis:
        """Analyze a structural design document using Gemini AI"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "analyze_design.txt")
            with open(prompt_path, "r") as f:
                prompt = f.read()

            filepath = pathlib.Path(design_file_path)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=filepath.read_bytes(),
                        mime_type='application/pdf',
                    ),
                    prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DesignAnalysis,
                },
            )
            if response.parsed:
                design_analysis = DesignAnalysis.model_validate(response.parsed)
                return design_analysis
            else:
                raw_json = response.candidates[0].content.parts[0].text
                try:
                    parsed_dict = json.loads(raw_json)
                    design_analysis = DesignAnalysis.model_validate(parsed_dict)
                    return design_analysis
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from raw text. {e}")
                    return DesignAnalysis(filename=None, city=None, project_type=None, design_description=None,
                                          key_elements=[])
                except Exception as e:
                    print(f"Error: Could not validate JSON against DesignAnalysis model. {e}")
                    return DesignAnalysis(filename=None, city=None, project_type=None, design_description=None,
                                          key_elements=[])

        except Exception as e:
            print(f"Error analyzing design document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing design document: {str(e)}")

    async def generate_structural_checklist(self, design_analysis: DesignAnalysis) -> ChecklistResponse:
        """Generate a contextual checklist based on design analysis"""
        relevant_comments = self.get_contextual_comments(design_analysis)
        comments_context = "\n".join([
            f"- {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''} (City: {doc.metadata.get('city', 'Unknown')}, Type: {doc.metadata.get('project_type', 'Unknown')})"
            for doc in relevant_comments[:30]
        ])

        prompt = f"""You are an expert structural engineer. Based on the provided design analysis, past city review comments, and the attached General Structural QA Checklist, generate a comprehensive, detailed, and to-the-point checklist in valid JSON format.

Prioritize and adapt items from the "General Structural QA Checklist" to be highly relevant to the specific project details and concerns raised in the city comments. Ensure the checklist is thorough and actionable.

DESIGN ANALYSIS:
- Project Type: {design_analysis.project_type}
- City/Location: {design_analysis.city}
- Design Description: {design_analysis.design_description}
- Key Elements: {', '.join(design_analysis.key_elements)}

RELEVANT PAST CITY COMMENTS ({len(relevant_comments)} comments from {design_analysis.city}):
{comments_context}

You must return ONLY a valid JSON object with this exact structure:
{{
  "project_info": {{
    "project_name": "{design_analysis.project_type} Project",
    "client_name": "TBD"
  }},
  "checklist_items": [
    {{
      "category": "Structural",
      "item": "Verify foundation design calculations based on geotechnical report recommendations.",
      "description": "Cross-reference proposed foundation sizing, reinforcement, and bearing capacity calculations with the specific recommendations for this project's soil conditions.",
      "priority": "High"
    }},
    {{
      "category": "Code Compliance",
      "item": "Review seismic design requirements for local building codes.",
      "description": "Ensure seismic design parameters and lateral force resisting system meet the latest building code standards relevant to the project's location.",
      "priority": "High"
    }}
  ],
  "relevant_comments_count": {len(relevant_comments)},
  "summary_of_key_concerns": "Summarize the overarching concerns derived from the relevant past city comments and their implications for the design.",
  "suggested_next_steps": [
    "Outline actionable next steps for the engineering team, prioritizing critical items based on the generated checklist and identified concerns."
  ]
}}

REQUIREMENTS:
1. Select, adapt, and refine checklist items making them specifically relevant to the design analysis and city comments.
2. Generate a comprehensive list of checklist items - include all highly relevant and specific checks.
3. Categories: Structural, Code Compliance, Documentation, Design Details, Safety.
4. Priority levels: High, Medium, Low.
5. Return ONLY the JSON object - no additional text or formatting."""

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ChecklistResponse,
                },
            )

            if response.parsed:
                checklist_data = ChecklistResponse.model_validate(response.parsed)
            else:
                raw_json = response.candidates[0].content.parts[0].text
                try:
                    parsed_dict = json.loads(raw_json)
                    checklist_data = ChecklistResponse.model_validate(parsed_dict)
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from raw text. {e}")
                    # Return fallback response
                    return ChecklistResponse(
                        project_info=ProjectInfo(project_name="Unknown Project", client_name="TBD"),
                        checklist_items=[],
                        relevant_comments_count=0,
                        summary_of_key_concerns="Unable to generate checklist",
                        suggested_next_steps=["Please try again"]
                    )
                except Exception as e:
                    print(f"Error: Could not validate JSON against ChecklistResponse model. {e}")
                    return ChecklistResponse(
                        project_info=ProjectInfo(project_name="Unknown Project", client_name="TBD"),
                        checklist_items=[],
                        relevant_comments_count=0,
                        summary_of_key_concerns="Unable to generate checklist",
                        suggested_next_steps=["Please try again"]
                    )

            return checklist_data

        except Exception as e:
            print(f"Error generating checklist with Gemini: {str(e)}")
            # Return fallback response
            return ChecklistResponse(
                project_info=ProjectInfo(project_name="Unknown Project", client_name="TBD"),
                checklist_items=[],
                relevant_comments_count=0,
                summary_of_key_concerns="Error generating checklist",
                suggested_next_steps=["Please try again"]
            )

    def get_contextual_comments(self, design_analysis: DesignAnalysis, max_comments: int = 50,
                                context_comments: int = 30) -> List[Document]:
        """Retrieve contextually relevant comments based on design analysis"""
        try:
            search_query = f"{design_analysis.design_description} {' '.join(design_analysis.key_elements)}"
            city_filter = design_analysis.city if design_analysis.city else None
            project_type_filter = design_analysis.project_type if design_analysis.project_type else None

            print(
                f"Searching for comments - City: {city_filter}, Project Type: {project_type_filter}, Max: {max_comments}")

            relevant_comments = self.search_similar_documents(
                query=search_query,
                limit=max_comments,
                city_filter=city_filter,
                project_type_filter=project_type_filter
            )

            print(
                f"Retrieved {len(relevant_comments)} relevant comments for city: {city_filter}, project type: {project_type_filter}")
            return relevant_comments

        except Exception as e:
            print(f"Error retrieving contextual comments: {str(e)}")
            return []

    def cleanup_embeddings(self):
        """Clean up embeddings from memory - MEMORY OPTIMIZATION"""
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
            cleanup_memory()
            print("Embeddings cleaned up from memory")


class MainService:
    def __init__(self):
        # Only initialize lightweight components at startup - MEMORY OPTIMIZATION
        self.plan_check = PlanCheck()

        # Don't load vectorstore at startup - MEMORY OPTIMIZATION
        self.vectorstore_loaded = False

        try:
            self.db = get_database()
            print("MainService initialized with database connection")
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            self.db = None

    def _ensure_vectorstore_loaded(self):
        """Load vectorstore only when needed - MEMORY OPTIMIZATION"""
        if not self.vectorstore_loaded:
            print("Loading vectorstore on demand...")
            self.vectorstore_loaded = self.plan_check.load_vectorstore()
            if self.vectorstore_loaded:
                self.plan_check.setup_retriever()
                print("Vectorstore loaded successfully")
            else:
                print("No existing vectorstore found - will create when needed")

    async def ingest_comments(self, temp_file_paths: List[str]) -> Dict:
        """Ingest comments - loads embeddings on demand"""
        try:
            print("Starting comment ingestion...")
            project_comments_data = await self.plan_check.extract_comments_with_gemini(temp_file_paths)
            await self.plan_check.process_comments(project_comments_data, append_mode=True)

            # Clean up embeddings after use - MEMORY OPTIMIZATION
            self.plan_check.cleanup_embeddings()

            self.vectorstore_loaded = True
            return {
                "status": "success",
                "message": f"Successfully ingested {len(project_comments_data)} comment files into Qdrant",
                "files_processed": len(project_comments_data)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error ingesting comments: {str(e)}",
                "files_processed": 0
            }

    #################### CRUD for users ####################
    async def google_sign_in(self, user_data: dict) -> str:
        """Create or update a Google user in MongoDB."""
        filter_query = {"email": user_data.get("email", "")}
        update_data = {
            "$set": {
                "full_name": user_data.get("full_name", ""),
                "email_verified": user_data.get("email_verified", False),
                "last_login_at": user_data.get("last_sign_in_at", datetime.now()),
                "role": user_data.get("role", "user"),
                "status": user_data.get("status", "active"),
                "photo_url": user_data.get("photo_url", ""),
            },
            "$setOnInsert": {
                "_id": user_data.get("firebase_uid", ""),
                "created_at": user_data.get("created_at", datetime.now()),
            }
        }

        updated_user = await self.db["users"].find_one_and_update(
            filter_query,
            update_data,
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        return str(updated_user["_id"])

    #################### CRUD for Checklist ####################
    async def generate_structural_checklist(self, file_path: str, user_id: str, checklist_id: str) -> ChecklistResponse:
        """Generate checklist - loads embeddings on demand"""
        try:
            # Ensure vectorstore is loaded before generating checklist
            self._ensure_vectorstore_loaded()

            structural_design_analysis = await self.plan_check.analyze_structural_design_with_gemini(file_path)
            structural_checklist = await self.plan_check.generate_structural_checklist(structural_design_analysis)

            # Clean up embeddings after use - MEMORY OPTIMIZATION
            self.plan_check.cleanup_embeddings()

            structural_checklist_dict = structural_checklist.model_dump()
            data = {
                "_id": checklist_id,
                "user_id": user_id,
                "project_info": structural_checklist_dict.get("project_info", {}),
                "checklist_items": structural_checklist_dict.get("checklist_items", []),
                "relevant_comments_count": structural_checklist_dict.get("relevant_comments_count", 0),
                "summary_of_key_concerns": structural_checklist_dict.get("summary_of_key_concerns", ""),
                "suggested_next_steps": structural_checklist_dict.get("suggested_next_steps", []),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }

            await self.db["checklists"].insert_one(data)
            return structural_checklist

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating design checklist: {str(e)}")

    async def get_checklists_by_user(self, user_id) -> List[dict]:
        """Get recent checklists"""
        cursor = self.db["checklists"].find({"user_id": user_id}).sort("created_at", -1)
        checklists = await cursor.to_list(None)
        for checklist in checklists:
            checklist["_id"] = str(checklist["_id"])
        return checklists

    async def get_checklist_by_id(self, checklist_id: str) -> Optional[dict]:
        """Get checklist by ID"""
        checklist = await self.db["checklists"].find_one({"_id": checklist_id})
        if checklist:
            checklist["_id"] = str(checklist["_id"])
        return checklist

    async def update_checklist(self, checklist_id: str, project_info: ProjectInfo) -> dict:
        result = await self.db["checklists"].update_one(
            {"_id": checklist_id},
            {"$set": {"project_info": project_info.model_dump()}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Checklist not found")
        updated = await self.db["checklists"].find_one({"_id": checklist_id})
        if updated:
            updated["_id"] = str(updated["_id"])
        return updated

    async def delete_checklist(self, checklist_id: str):
        result = await self.db["checklists"].delete_one({"_id": checklist_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Checklist not found")
        return {"message": "Checklist deleted successfully"}

    #################### CRUD for Plan Check ####################
    async def execute_plan_check(self, temp_file_paths: List[str], user_id: str,
                                 plan_check_id: str) -> PlanCheckResponse:
        """Execute plan check on the provided design set"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "plan_check.txt")
        with open(prompt_path, "r") as f:
            prompt = f.read()

        try:
            uploaded_files = []
            for pdf_path in temp_file_paths:
                try:
                    uploaded_file = client.files.upload(file=pdf_path)
                    uploaded_files.append(uploaded_file)
                except Exception as e:
                    print(f"Error uploading {pdf_path} to Gemini Files API: {e}")

            contents = [*uploaded_files, prompt]
            response = self.plan_check.gemini_client.models.generate_content(
                contents=contents,
                model="gemini-2.5-flash",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PlanCheckResponse
                }
            )

            plan_check_response = None
            if response.parsed:
                plan_check_response = PlanCheckResponse.model_validate(response.parsed)
            else:
                raw_json = response.candidates[0].content.parts[0].text
                try:
                    parsed_dict = json.loads(raw_json)
                    plan_check_response = PlanCheckResponse.model_validate(parsed_dict)
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from raw text. {e}")
                    return PlanCheckResponse(project_info=ProjectInfo(project_name="Unknown", client_name="Unknown"),
                                             items=[])
                except Exception as e:
                    print(f"Error: Could not validate JSON against PlanCheckResponse model. {e}")
                    return PlanCheckResponse(project_info=ProjectInfo(project_name="Unknown", client_name="Unknown"),
                                             items=[])

            plan_check_response_dict = plan_check_response.model_dump()

            data = {
                "_id": plan_check_id,
                "user_id": user_id,
                "project_info": plan_check_response_dict.get("project_info", {}),
                "items": plan_check_response_dict.get("items", []),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }

            await self.db["plan-reviews"].insert_one(data)
            return plan_check_response

        except Exception as e:
            print(f"Error executing plan check: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error executing plan check: {str(e)}")

    async def get_plan_checks_by_user(self, user_id) -> List[dict]:
        """Get recent plan checks"""
        cursor = self.db["plan-reviews"].find({"user_id": user_id}).sort("created_at", -1)
        plan_checks = await cursor.to_list(None)
        for plan_check in plan_checks:
            plan_check["_id"] = str(plan_check["_id"])
        return plan_checks

    async def get_plan_check_by_id(self, plan_check_id: str) -> Optional[dict]:
        """Get plan check by ID"""
        plan_check = await self.db["plan-reviews"].find_one({"_id": plan_check_id})
        if plan_check:
            plan_check["_id"] = str(plan_check["_id"])
        return plan_check

    async def update_plan_check(self, plan_check_id: str, project_info: ProjectInfo) -> dict:
        result = await self.db["plan-reviews"].update_one(
            {"_id": plan_check_id},
            {"$set": {"project_info": project_info.model_dump()}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="plan check not found")
        updated = await self.db["plan-reviews"].find_one({"_id": plan_check_id})
        if updated:
            updated["_id"] = str(updated["_id"])
        return updated