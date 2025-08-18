# Core libraries
import datetime
import json
import os
from typing import List, Dict, Optional, Any
import pathlib
import gc
import hashlib

from bson import ObjectId
from datetime import datetime
# FastAPI:
from fastapi import HTTPException

# Pydantic
from pydantic import BaseModel

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
import re
import math
from pymongo import DESCENDING

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


class QAResponse(BaseModel):
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
    state: str
    project_type: str
    design_description: str
    key_elements: List[str]


# City Comments:
class Comment(BaseModel):
    text: str


class ProjectCommentsResponse(BaseModel):
    filename: str
    city: str
    state: str
    project_type: str
    comments: List[Comment]


class MonthlyData(BaseModel):
    month: str
    count: int


class DashboardStats(BaseModel):
    total_projects_this_month: int
    active_projects_this_month: int
    completed_projects: int
    projects_by_location: List[Dict[str, Any]]
    top_issue_categories: List[Dict[str, Any]]
    monthly_completed_projects: List[MonthlyData]


def cleanup_memory():
    """Force garbage collection to free up memory"""
    collected = gc.collect()
    print(f"Garbage collector: collected {collected} objects")


class PlanCheck:
    def __init__(self):
        self.gemini_client = client
        self.qdrant_client = None
        self.default_collection = "city_comments"  # Default collection name

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

    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content to detect duplicates"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _check_for_duplicates(self, chunks: List[Document], collection_name: str) -> List[Document]:
        """Remove duplicate documents based on content hash"""
        try:
            print(f"Checking for duplicates in {len(chunks)} documents...")

            # Generate hashes for new content
            new_content_hashes = set()
            unique_chunks = []

            for chunk in chunks:
                content_hash = self._generate_content_hash(chunk.page_content)
                if content_hash not in new_content_hashes:
                    new_content_hashes.add(content_hash)
                    unique_chunks.append(chunk)

            print(f"Removed {len(chunks) - len(unique_chunks)} duplicate documents from current batch")

            # Check against existing documents in the collection
            if self._collection_exists(collection_name):
                try:
                    # Search for existing documents with similar content
                    embeddings = self._get_embeddings()
                    existing_hashes = set()

                    # Sample a few documents to check for existing hashes
                    # This is a simplified approach - in production, you might want to store hashes as metadata
                    for chunk in unique_chunks[:10]:  # Check first 10 documents as sample
                        query_embedding = embeddings.embed_query(chunk.page_content)
                        search_results = self.qdrant_client.search(
                            collection_name=collection_name,
                            query_vector=query_embedding,
                            limit=5,
                            score_threshold=0.98,  # Very high threshold for near-exact matches
                            with_payload=True
                        )

                        for result in search_results:
                            existing_content = result.payload.get("content", "")
                            existing_hash = self._generate_content_hash(existing_content)
                            existing_hashes.add(existing_hash)

                    # Filter out documents that already exist
                    final_unique_chunks = []
                    for chunk in unique_chunks:
                        content_hash = self._generate_content_hash(chunk.page_content)
                        if content_hash not in existing_hashes:
                            final_unique_chunks.append(chunk)

                    print(
                        f"Removed {len(unique_chunks) - len(final_unique_chunks)} documents that already exist in collection")
                    return final_unique_chunks

                except Exception as e:
                    print(f"Warning: Could not check for existing duplicates: {e}")
                    return unique_chunks

            return unique_chunks

        except Exception as e:
            print(f"Error in duplicate checking: {e}")
            return chunks  # Return original chunks if duplicate checking fails

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        try:
            self.qdrant_client.get_collection(collection_name)
            return True
        except:
            return False

    def create_or_update_vectorstore(self, chunks: List[Document], collection_name: str = None,
                                     append_mode: bool = True):
        """Create or append to vector store - loads embeddings on demand"""
        if collection_name is None:
            collection_name = self.default_collection

        embeddings = self._get_embeddings()  # Load embeddings only when needed

        # Check for duplicates
        unique_chunks = self._check_for_duplicates(chunks, collection_name)

        if not unique_chunks:
            print("No new unique documents to add after duplicate removal")
            return

        if append_mode:
            print(
                f"Appending {len(unique_chunks)} new unique documents to existing vector store '{collection_name}'...")
        else:
            print(f"Creating new vector store '{collection_name}' with {len(unique_chunks)} documents...")
            try:
                self.qdrant_client.delete_collection(collection_name)
                print(f"Deleted existing collection '{collection_name}'")
            except:
                pass  # Collection might not exist

        # Check if collection exists, create if needed
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            print(f"Collection '{collection_name}' exists with {collection_info.vectors_count} vectors")
        except Exception:
            # Collection doesn't exist, create it
            print(f"Creating collection '{collection_name}'...")
            sample_embedding = embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="city",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="state",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="project_type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="file_name",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content_hash",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"Created collection '{collection_name}' with dimension {embedding_dim} and payload indexes")
            except Exception as e:
                print(f"Note: Some indexes may already exist: {e}")

        try:
            points = []
            for i, chunk in enumerate(unique_chunks):
                embedding = embeddings.embed_query(chunk.page_content)
                content_hash = self._generate_content_hash(chunk.page_content)

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk.page_content,
                        "city": chunk.metadata.get("city", "Unknown"),
                        "state": chunk.metadata.get("state", "Unknown"),
                        "project_type": chunk.metadata.get("project_type", "Unknown"),
                        "file_name": chunk.metadata.get("file_name", "Unknown"),
                        "page": chunk.metadata.get("page", None),
                        "content_hash": content_hash  # Store hash for future duplicate detection
                    }
                )
                points.append(point)

            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

            collection_info = self.qdrant_client.get_collection(collection_name)
            action = "appended to" if append_mode else "created in"
            print(
                f"Successfully {action} vector store '{collection_name}'. Total vectors: {collection_info.vectors_count}")

        except Exception as e:
            print(f"Error creating/updating vector store: {str(e)}")
            raise

    def load_vectorstore(self, collection_name: str = None):
        """Check if vector store exists - lightweight operation"""
        if collection_name is None:
            collection_name = self.default_collection

        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            print(f"Found existing Qdrant collection '{collection_name}': {collection_info.vectors_count} vectors")
            return True
        except Exception as e:
            print(f"No existing vectorstore '{collection_name}' found - will create when needed")
            return False

    def _search_with_filters(self, search_query: str, max_comments: int,
                             collection_name: str = None,
                             city_filter: Optional[str] = None,
                             state_filter: Optional[str] = None,
                             project_type_filter: Optional[str] = None) -> List[Document]:
        """Helper method to search with specific filters"""
        if collection_name is None:
            collection_name = self.default_collection

        try:
            embeddings = self._get_embeddings()
            query_embedding = embeddings.embed_query(search_query)

            filter_conditions = []

            # Add city filter if provided
            if city_filter:
                filter_conditions.append(
                    FieldCondition(key="city", match=MatchValue(value=city_filter))
                )

            # Add state filter if provided
            if state_filter:
                filter_conditions.append(
                    FieldCondition(key="state", match=MatchValue(value=state_filter))
                )

            # Add project type filter if provided
            if project_type_filter:
                filter_conditions.append(
                    FieldCondition(key="project_type", match=MatchValue(value=project_type_filter))
                )

            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)

            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=max_comments,
                with_payload=True
            )

            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload["content"],
                    metadata={
                        "city": result.payload.get("city", "Unknown"),
                        "state": result.payload.get("state", "Unknown"),
                        "project_type": result.payload.get("project_type", "Unknown"),
                        "file_name": result.payload.get("file_name", "Unknown"),
                        "page": result.payload.get("page", None),
                        "score": result.score,
                        "content_hash": result.payload.get("content_hash", "")
                    }
                )
                documents.append(doc)

            filter_desc = []
            if city_filter:
                filter_desc.append(f"City: {city_filter}")
            if state_filter:
                filter_desc.append(f"State: {state_filter}")
            if project_type_filter:
                filter_desc.append(f"Project Type: {project_type_filter}")

            print(
                f"Search in '{collection_name}' with filters [{', '.join(filter_desc)}] returned {len(documents)} results")
            return documents

        except Exception as e:
            print(f"Error in filtered search: {str(e)}")
            return []

    def setup_retriever(self, collection_name: str = None):
        """Lightweight setup - just check collection exists"""
        if collection_name is None:
            collection_name = self.default_collection

        print(f"Setting up retriever for collection '{collection_name}'...")
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            print(f"Qdrant collection '{collection_name}' ready with {collection_info.vectors_count} vectors")
            return True
        except Exception as e:
            print(f"Collection '{collection_name}' doesn't exist yet - will be created when needed")
            return False

    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            print(f"Available collections: {collection_names}")
            return collection_names
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    async def process_comments(self, city_comments: List[ProjectCommentsResponse], collection_name: str = None,
                               append_mode: bool = True):
        """Process comments and create vectorstore"""
        if collection_name is None:
            collection_name = self.default_collection

        all_documents = []
        for city_comment in city_comments:
            try:
                for comment in city_comment.comments:
                    doc = Document(
                        page_content=comment.text,
                        metadata={
                            "city": city_comment.city,
                            "state": city_comment.state,
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
            self.create_or_update_vectorstore(all_documents, collection_name, append_mode)
            setup_success = self.setup_retriever(collection_name)
            if setup_success:
                action = "appended to" if append_mode else "created"
                print(f"Vector store '{collection_name}' {action} successfully!")
            else:
                print(f"Warning: Vector store '{collection_name}' updated but retriever setup failed")
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
                    return DesignAnalysis(filename=None, city=None, state=None, project_type=None,
                                          design_description=None,
                                          key_elements=[])
                except Exception as e:
                    print(f"Error: Could not validate JSON against DesignAnalysis model. {e}")
                    return DesignAnalysis(filename=None, city=None, state=None, project_type=None,
                                          design_description=None,
                                          key_elements=[])

        except Exception as e:
            print(f"Error analyzing design document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing design document: {str(e)}")

    async def generate_structural_checklist(self, design_analysis: DesignAnalysis, state: str, city: str,
                                            collection_name: str = None) -> ChecklistResponse:
        """Generate a contextual checklist based on design analysis"""
        if collection_name is None:
            collection_name = self.default_collection

        relevant_comments = self.get_contextual_comments(design_analysis, state, city, collection_name=collection_name)
        comments_context = "\n".join([
            f"- {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''} (City: {doc.metadata.get('city', 'Unknown')}, State: {doc.metadata.get('state', 'Unknown')}, Type: {doc.metadata.get('project_type', 'Unknown')})"
            for doc in relevant_comments[:30]
        ])

        prompt = f"""You are an expert structural engineer. Based on the provided design analysis, past city review comments, and the attached General Structural QA Checklist, generate a comprehensive, detailed, and to-the-point checklist in valid JSON format.

    Prioritize and adapt items from the "General Structural QA Checklist" to be highly relevant to the specific project details and concerns raised in the city comments. Ensure the checklist is thorough and actionable.

    DESIGN ANALYSIS:
    - Project Type: {design_analysis.project_type}
    - CityLocation: {city}, {state}
    - Design Description: {design_analysis.design_description}
    - Key Elements: {', '.join(design_analysis.key_elements)}

    RELEVANT PAST CITY COMMENTS ({len(relevant_comments)} comments from {city}, {state} using collection '{collection_name}'):
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

            checklist_data = None
            if response.parsed:
                checklist_data = ChecklistResponse.model_validate(response.parsed)
            else:
                raw_json = response.candidates[0].content.parts[0].text
                try:
                    parsed_dict = json.loads(raw_json)
                    checklist_data = ChecklistResponse.model_validate(parsed_dict)
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from raw text. {e}")
                    # Create fallback response instead of returning
                    checklist_data = ChecklistResponse(
                        project_info=ProjectInfo(project_name="Unknown Project", client_name="TBD"),
                        checklist_items=[],
                        relevant_comments_count=0,
                        summary_of_key_concerns="Unable to generate checklist due to JSON parsing error",
                        suggested_next_steps=["Please try again or contact support"]
                    )
                except Exception as e:
                    print(f"Error: Could not validate JSON against ChecklistResponse model. {e}")
                    # Create fallback response instead of returning
                    checklist_data = ChecklistResponse(
                        project_info=ProjectInfo(project_name="Unknown Project", client_name="TBD"),
                        checklist_items=[],
                        relevant_comments_count=0,
                        summary_of_key_concerns="Unable to generate checklist due to validation error",
                        suggested_next_steps=["Please try again or contact support"]
                    )

            return checklist_data

        except Exception as e:
            print(f"Error generating checklist with Gemini: {str(e)}")
            # Return fallback response instead of raising
            return ChecklistResponse(
                project_info=ProjectInfo(project_name="Unknown Project", client_name="TBD"),
                checklist_items=[],
                relevant_comments_count=0,
                summary_of_key_concerns=f"Error generating checklist: {str(e)}",
                suggested_next_steps=["Please try again or contact support"]
            )

    def get_contextual_comments(self, design_analysis: DesignAnalysis, state: str, city: str,
                                collection_name: str = None, max_comments: int = 50) -> List[Document]:
        """Retrieve contextually relevant comments based on design analysis with hierarchical search strategy"""
        if collection_name is None:
            collection_name = self.default_collection

        try:
            search_query = f"{design_analysis.design_description} {' '.join(design_analysis.key_elements)}"
            city_filter = city
            state_filter = state
            project_type_filter = design_analysis.project_type if design_analysis.project_type else None

            print(
                f"Starting hierarchical search in '{collection_name}' - City: {city_filter}, State: {state_filter}, Project Type: {project_type_filter}")

            # Level 1: Search by City, State, and Project Type
            if city_filter and state_filter and project_type_filter:
                print("Level 1: Searching by City, State, and Project Type")
                relevant_comments = self._search_with_filters(
                    search_query, max_comments, collection_name, city_filter, state_filter, project_type_filter
                )
                if len(relevant_comments) > 0:
                    print(f"Level 1 successful: Found {len(relevant_comments)} comments")
                    return relevant_comments

            # Level 2: Search by State and Project Type only
            if state_filter and project_type_filter:
                print("Level 2: Searching by State and Project Type only")
                relevant_comments = self._search_with_filters(
                    search_query, max_comments, collection_name, None, state_filter, project_type_filter
                )
                if len(relevant_comments) > 0:
                    print(f"Level 2 successful: Found {len(relevant_comments)} comments")
                    return relevant_comments

            # Level 3: Search by State only
            if state_filter:
                print("Level 3: Searching by State only")
                relevant_comments = self._search_with_filters(
                    search_query, max_comments, collection_name, None, state_filter, None
                )
                if len(relevant_comments) > 0:
                    print(f"Level 3 successful: Found {len(relevant_comments)} comments")
                    return relevant_comments

            # Level 4: No filters - let LLM do the heavy lifting
            print("Level 4: No relevant comments found, returning empty list for LLM processing")
            return []

        except Exception as e:
            print(f"Error retrieving contextual comments from '{collection_name}': {str(e)}")
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
        self.vectorstore_loaded = {}  # Track loaded collections

        try:
            self.db = get_database()
            print("MainService initialized with database connection")
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            self.db = None

    def _ensure_vectorstore_loaded(self, collection_name: str = None):
        """Load vectorstore only when needed - MEMORY OPTIMIZATION"""
        if collection_name is None:
            collection_name = self.plan_check.default_collection

        if collection_name not in self.vectorstore_loaded:
            print(f"Loading vectorstore '{collection_name}' on demand...")
            self.vectorstore_loaded[collection_name] = self.plan_check.load_vectorstore(collection_name)
            if self.vectorstore_loaded[collection_name]:
                self.plan_check.setup_retriever(collection_name)
                print(f"Vectorstore '{collection_name}' loaded successfully")
            else:
                print(f"No existing vectorstore '{collection_name}' found - will create when needed")

    def list_available_collections(self) -> List[str]:
        """List all available collections"""
        return self.plan_check.list_collections()

    async def ingest_comments(self, temp_file_paths: List[str], collection_name: str = "city_comments") -> Dict:
        """Ingest comments - loads embeddings on demand"""
        try:
            print(f"Starting comment ingestion into collection '{collection_name}'...")
            project_comments_data = await self.plan_check.extract_comments_with_gemini(temp_file_paths)
            await self.plan_check.process_comments(project_comments_data, collection_name, append_mode=True)

            # Clean up embeddings after use - MEMORY OPTIMIZATION
            self.plan_check.cleanup_embeddings()

            self.vectorstore_loaded[collection_name] = True
            return {
                "status": "success",
                "message": f"Successfully ingested {len(project_comments_data)} comment files into collection '{collection_name}'",
                "files_processed": len(project_comments_data),
                "collection_name": collection_name
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error ingesting comments into '{collection_name}': {str(e)}",
                "files_processed": 0,
                "collection_name": collection_name
            }

    # Users Business Logic:
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

    # Checklist Business Logic:
    async def generate_structural_checklist(self, file_path: str, user_id: str, project_id: str, title: str, state: str,
                                            city: str, collection_name: str = "city_comments"):
        """Generate checklist - loads embeddings on demand"""
        try:
            # Ensure vectorstore is loaded before generating checklist
            self._ensure_vectorstore_loaded(collection_name)

            structural_design_analysis = await self.plan_check.analyze_structural_design_with_gemini(file_path)
            structural_checklist = await self.plan_check.generate_structural_checklist(
                structural_design_analysis, state, city, collection_name
            )

            # Clean up embeddings after use - MEMORY OPTIMIZATION
            self.plan_check.cleanup_embeddings()

            # Convert to dict for database storage
            structural_checklist_dict = structural_checklist.model_dump()

            data = {
                "user_id": user_id,
                "project_id": ObjectId(project_id),
                "title": title,
                "collection_used": collection_name,  # Track which collection was used
                "project_info": structural_checklist_dict.get("project_info", {}),  # Store project_info
                "checklist_items": structural_checklist_dict.get("checklist_items", []),
                "checklist_item_count": len(structural_checklist_dict.get("checklist_items", [])),
                "relevant_comments_count": structural_checklist_dict.get("relevant_comments_count", 0),
                "summary_of_key_concerns": structural_checklist_dict.get("summary_of_key_concerns", ""),
                "suggested_next_steps": structural_checklist_dict.get("suggested_next_steps", []),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }

            result = await self.db["checklists"].insert_one(data)

            await self.db["projects"].update_one(
                {"_id": ObjectId(project_id), "user_id": user_id},
                {"$inc": {"checklist_count": 1}}
            )

            return {
                "_id": str(result.inserted_id),
                "collection_used": collection_name,
                **structural_checklist.model_dump(),
            }

        except Exception as e:
            print(f"Error generating design checklist: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating design checklist: {str(e)}")

    async def get_checklist_by_id(self, checklist_id: str):
        """Get checklist by ID"""
        checklist = await self.db["checklists"].find_one({"_id": ObjectId(checklist_id)})
        if checklist:
            checklist["_id"] = str(checklist["_id"])
            checklist["project_id"] = str(checklist["project_id"])
        return checklist

    async def delete_checklist(self, checklist_id: str):
        result = await self.db["checklists"].delete_one({"_id": ObjectId(checklist_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Checklist not found")
        return {"message": "Checklist deleted successfully"}

    # QA Business Logic:
    async def execute_qa(self, temp_file_paths: List[str], user_id: str,
                         project_id: str, title: str):
        """Execute QA on the provided design set"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "plan_qa.txt")
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
                    "response_schema": QAResponse
                }
            )

            qa_response = None
            if response.parsed:
                qa_response = QAResponse.model_validate(response.parsed)
            else:
                raw_json = response.candidates[0].content.parts[0].text
                try:
                    parsed_dict = json.loads(raw_json)
                    qa_response = QAResponse.model_validate(parsed_dict)
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from raw text. {e}")
                    # Create a proper QAResponse object instead of returning it
                    qa_response = QAResponse(
                        project_info=ProjectInfo(project_name="Unknown", client_name="Unknown"),
                        items=[]
                    )
                except Exception as e:
                    print(f"Error: Could not validate JSON against QAResponse model. {e}")
                    # Create a proper QAResponse object instead of returning it
                    qa_response = QAResponse(
                        project_info=ProjectInfo(project_name="Unknown", client_name="Unknown"),
                        items=[]
                    )

            # Now qa_response is guaranteed to be a QAResponse object
            qa_response_dict = qa_response.model_dump()

            data = {
                "title": title,
                "project_id": ObjectId(project_id),
                "user_id": user_id,
                "project_info": qa_response_dict.get("project_info", {}),  # Add this line
                "items": qa_response_dict.get("items", []),
                "qa_item_count": len(qa_response_dict.get("items", [])),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }

            result = await self.db["qas"].insert_one(data)

            await self.db["projects"].update_one(
                {"_id": ObjectId(project_id), "user_id": user_id},
                {"$inc": {"qa_count": 1}}
            )

            return {
                "_id": str(result.inserted_id),
                **qa_response.model_dump(),
            }

        except Exception as e:
            print(f"Error executing plan check: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error executing plan check: {str(e)}")

    async def get_qa_by_id(self, qa_id: str) -> Optional[dict]:
        """Get QA by ID"""
        qa = await self.db["qas"].find_one({"_id": ObjectId(qa_id)})
        if qa:
            qa["_id"] = str(qa["_id"])
            qa["project_id"] = str(qa["project_id"])
        return qa

    async def delete_qa(self, qa_id: str):
        """Delete QA by ID"""
        result = await self.db["qas"].delete_one({"_id": ObjectId(qa_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="QA not found")
        return {"message": "QA deleted successfully"}

    # Project Business Logic:
    async def create_project(self, project_data: dict) -> dict:
        """Create a new project for a user"""
        try:
            data = {
                "project_name": project_data.get("project_name", ""),
                "client_name": project_data.get("client_name", ""),
                "project_type": project_data.get("project_type", ""),
                "state": project_data.get("state", ""),
                "city": project_data.get("city", ""),
                "user_id": project_data.get("user_id", ""),
                "checklist_count": project_data.get("checklist_count", 0),
                "qa_count": project_data.get("qa_count", 0),
                "created_at": datetime.now(),
                "domain": project_data.get("domain", "structural"),
                "status": project_data.get("status", "in_progress"),
            }

            result = await self.db["projects"].insert_one(data)

            return {"_id": str(result.inserted_id)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating a new project: {str(e)}")

    async def get_projects_by_user(
            self,
            user_id: str,
            page: int = 1,
            limit: int = 4,
            search: str = "",
            status: Optional[str] = None
    ) -> dict:
        """Get paginated projects with search/filter - NO BACKEND CACHING"""

        # Build filter query
        filter_query = {"user_id": user_id}

        # Add search filter
        if search and search.strip():
            search_regex = re.compile(re.escape(search.strip()), re.IGNORECASE)
            filter_query["$or"] = [
                {"project_name": {"$regex": search_regex}},
                {"client_name": {"$regex": search_regex}},
                {"city": {"$regex": search_regex}},
                {"state": {"$regex": search_regex}},
                {"project_type": {"$regex": search_regex}}
            ]

        # Add status filter
        if status:
            filter_query["status"] = status

        # Count total and calculate pagination
        total_projects = await self.db["projects"].count_documents(filter_query)
        total_pages = math.ceil(total_projects / limit) if total_projects > 0 else 1
        skip = (page - 1) * limit

        # Fetch projects (NO CACHING HERE)
        cursor = self.db["projects"].find(filter_query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        projects = await cursor.to_list(None)

        # Convert ObjectId to string
        for project in projects:
            project["_id"] = str(project["_id"])

        return {
            "projects": projects,
            "total_projects": total_projects,
            "total_pages": total_pages,
            "current_page": page,
            "limit": limit
        }

    async def get_project_by_id(self, project_id) -> dict:
        """Get project by ID"""
        project = await self.db["projects"].find_one({"_id": ObjectId(project_id)})
        if project:
            project["_id"] = str(project["_id"])
        return project

    async def get_project_qas(self, project_id) -> List[dict]:
        """Get all QAs for a project"""
        try:
            project_obj_id = ObjectId(project_id)
        except:
            raise ValueError(f"Invalid project_id: {project_id}")

        cursor = self.db["qas"].find(
            {"project_id": project_obj_id}
        ).sort("created_at", -1)

        qas = await cursor.to_list(length=None)

        # Convert ObjectId to string for JSON serialization
        for qa in qas:
            qa["_id"] = str(qa["_id"])
            # Optional: convert project_id to string if you return it
            qa["project_id"] = str(qa["project_id"])
        return qas

    async def get_project_checklists(self, project_id) -> List[dict]:
        """Get all Checklists for a project"""
        try:
            project_obj_id = ObjectId(project_id)
        except:
            raise ValueError(f"Invalid project_id: {project_id}")

        cursor = self.db["checklists"].find(
            {"project_id": project_obj_id}
        ).sort("created_at", -1)

        checklists = await cursor.to_list(length=None)

        # Convert ObjectId to string for JSON serialization
        for checklist in checklists:
            checklist["_id"] = str(checklist["_id"])
            # Optional: convert project_id to string if you return it
            checklist["project_id"] = str(checklist["project_id"])
        return checklists

    async def delete_project(self, project_id: str, user_id: str):
        """Delete project and all associated checklists and QAs"""
        try:
            project_obj_id = ObjectId(project_id)

            # First verify the project belongs to the user
            project = await self.db["projects"].find_one({"_id": project_obj_id, "user_id": user_id})
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")

            # Delete associated checklists
            checklist_result = await self.db["checklists"].delete_many({"project_id": project_obj_id})

            # Delete associated QAs
            qa_result = await self.db["qas"].delete_many({"project_id": project_obj_id})

            # Delete the project
            project_result = await self.db["projects"].delete_one({"_id": project_obj_id, "user_id": user_id})

            if project_result.deleted_count == 0:
                raise HTTPException(status_code=404, detail="Project not found")

            return {
                "message": "Project and all associated data deleted successfully",
                "deleted_checklists": checklist_result.deleted_count,
                "deleted_qas": qa_result.deleted_count
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")

    # Logic for Dashboard:
    async def get_dashboard_stats(self, user_id: str) -> DashboardStats:
        """Get dashboard statistics for a user"""
        try:
            # Get current month and year date ranges
            now = datetime.now()
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            start_of_year = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

            # 1. Total Projects This Month
            total_projects_this_month = await self.db["projects"].count_documents({
                "user_id": user_id,
                "created_at": {"$gte": start_of_month}
            })

            # 2. Active Projects This Month
            active_projects_this_month = await self.db["projects"].count_documents({
                "user_id": user_id,
                "created_at": {"$gte": start_of_month},
                "status": "in_progress"
            })

            # 3. Completed Projects (all time)
            completed_projects = await self.db["projects"].count_documents({
                "user_id": user_id,
                "status": "completed"
            })

            # 4. Projects By Location
            location_pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$group": {
                        "_id": {
                            "city": "$city",
                            "state": "$state"
                        },
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "location": {
                            "$concat": [
                                {"$ifNull": ["$_id.city", "Unknown City"]},
                                ", ",
                                {"$ifNull": ["$_id.state", "Unknown State"]}
                            ]
                        },
                        "city": "$_id.city",
                        "state": "$_id.state",
                        "count": 1
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]

            location_cursor = self.db["projects"].aggregate(location_pipeline)
            projects_by_location = await location_cursor.to_list(length=None)

            # 5. Monthly Completed Projects This Year
            monthly_completed_pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "status": "completed",
                        "completed_at": {"$gte": start_of_year}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "year": {"$year": "$completed_at"},
                            "month": {"$month": "$completed_at"}
                        },
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "month": "$_id.month",
                        "year": "$_id.year",
                        "count": 1
                    }
                },
                {"$sort": {"month": 1}}
            ]

            monthly_cursor = self.db["projects"].aggregate(monthly_completed_pipeline)
            monthly_data = await monthly_cursor.to_list(length=None)

            # Fill in missing months with 0 counts
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            monthly_completed_projects = []
            existing_months = {item['month']: item['count'] for item in monthly_data}

            for month in range(1, 13):
                monthly_completed_projects.append({
                    "month": month_names[month - 1],
                    "count": existing_months.get(month, 0)
                })

            # 6. Top Issue Categories in QA Runs
            categories_pipeline = [
                {"$match": {"user_id": user_id}},
                {"$unwind": "$items"},
                {
                    "$group": {
                        "_id": "$items.category",
                        "count": {"$sum": 1},
                        "high_priority_count": {
                            "$sum": {
                                "$cond": [{"$eq": ["$items.priority", "High"]}, 1, 0]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "category": "$_id",
                        "count": 1,
                        "high_priority_count": 1
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": 8}
            ]

            categories_cursor = self.db["qas"].aggregate(categories_pipeline)
            top_issue_categories = await categories_cursor.to_list(length=None)

            # Calculate percentages
            total_items = sum(item['count'] for item in top_issue_categories) if top_issue_categories else 1
            for item in top_issue_categories:
                item['percentage'] = round((item['count'] / total_items) * 100, 1)

            return DashboardStats(
                total_projects_this_month=total_projects_this_month,
                active_projects_this_month=active_projects_this_month,
                completed_projects=completed_projects,
                projects_by_location=projects_by_location,
                top_issue_categories=top_issue_categories,
                monthly_completed_projects=monthly_completed_projects
            )

        except Exception as e:
            print(f"Error getting dashboard stats: {str(e)}")
            # Return empty stats on error for production stability
            return DashboardStats(
                total_projects_this_month=0,
                active_projects_this_month=0,
                completed_projects=0,
                projects_by_location=[],
                top_issue_categories=[],
                monthly_completed_projects=[]
            )