# Core libraries
import json
import os
from typing import List, Dict
import pathlib

# FastAPI:
from fastapi import HTTPException

# Pydantic
from pydantic import BaseModel

# LangChain:
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# Qdrant:
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

# Gemini:
from google import genai
from google.genai import types

# Load Environment variable
from dotenv import load_dotenv
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

class PlanCheck:
    def __init__(self):
        self.gemini_client = client

        self.qdrant_client = None
        self.collection_name = "city_comments"

        # Initialize embeddings
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Initialize Qdrant client
        self._initialize_qdrant()

    # Vector DB Logic:
    def _initialize_qdrant(self):
        """Initialize Qdrant cloud client with production-ready error handling"""
        try:
            # Get Qdrant cloud credentials from environment
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if not qdrant_url or not qdrant_api_key:
                raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables are required")

            # Initialize client with timeout and retry settings for production
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30,  # 30 second timeout for production
            )

            # Todo: Remove Test Connection
            # Test connection
            collections = self.qdrant_client.get_collections()
            print(f"Successfully connected to Qdrant cloud. Available collections: {len(collections.collections)}")

            # Todo: Remove Sample Embedding
            # Get embedding dimension
            sample_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)

            # Create collection if it doesn't exist, or ensure indexes exist
            from qdrant_client.models import PayloadSchemaType

            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' exists with {collection_info.vectors_count} vectors")

                # Check if indexes exist and create them if they don't
                try:
                    # Try to create indexes (this will fail silently if they already exist)
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="city",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    print("Created city index")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("City index already exists")
                    else:
                        print(f"Error creating city index: {e}")

                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="project_type",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    print("Created project_type index")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("Project_type index already exists")
                    else:
                        print(f"Error creating project_type index: {e}")

                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="file_name",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    print("Created file_name index")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("File_name index already exists")
                    else:
                        print(f"Error creating file_name index: {e}")

            except Exception:
                print(f"Creating collection '{self.collection_name}'...")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )

                # Create payload indexes for filtering
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
            print(f"CRITICAL ERROR: Failed to initialize Qdrant connection: {str(e)}")
            raise RuntimeError(f"Cannot start service without Qdrant connection: {str(e)}")


    def create_or_update_vectorstore(self, chunks: List[Document], append_mode: bool = True):
        """
        Create or append to vector store from document chunks using Qdrant.
        Args:
            chunks: List of document chunks
            append_mode: If True, append to existing collection. If False, recreate collection.
        """
        if append_mode:
            print(f"Appending {len(chunks)} new documents to existing vector store...")
        else:
            print(f"Creating new vector store with {len(chunks)} documents...")
            # Delete and recreate collection if not in append mode
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                print(f"Deleted existing collection '{self.collection_name}'")
            except:
                pass  # Collection might not exist

            # Recreate collection
            sample_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

        try:
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = self.embeddings.embed_query(chunk.page_content)

                # Create point with metadata
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
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

            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            # Get updated count
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            action = "appended to" if append_mode else "created in"
            print(f"Successfully {action} vector store. Total vectors: {collection_info.vectors_count}")

        except Exception as e:
            print(f"Error creating/updating vector store: {str(e)}")
            raise

    def load_vectorstore(self):
        """
        Load existing vector store from Qdrant cloud.
        This is mainly for compatibility - Qdrant cloud is always available.
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Loaded existing Qdrant collection: {collection_info.vectors_count} vectors")
            return True
        except Exception as e:
            print(f"Error loading vectorstore: {str(e)}")
            return False

    def search_similar_documents(self, query: str, limit: int = 20, city_filter: str = None,
                                 project_type_filter: str = None) -> List[Document]:
        """
        Search for similar documents in Qdrant.
        Args:
            query: Search query text
            limit: Maximum number of results to return
            city_filter: Optional city filter
            project_type_filter: Optional project type filter
        Returns:
            List of Document objects
        """
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)

            # DEBUG: First, let's see what data we actually have
            print("=== DEBUGGING SEARCH ===")

            # Get a sample of all data without filters to see what's stored
            all_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=5,  # Just get a few samples
                with_payload=True
            )

            print(f"Total collection has data: {len(all_results) > 0}")
            if all_results:
                print("Sample data in collection:")
                for i, result in enumerate(all_results[:3]):
                    city = result.payload.get("city", "NO_CITY")
                    proj_type = result.payload.get("project_type", "NO_PROJECT_TYPE")
                    print(f"  Sample {i + 1}: City='{city}', ProjectType='{proj_type}'")

            print(f"Looking for: City='{city_filter}', ProjectType='{project_type_filter}'")

            # Now try the filtered search
            filter_conditions = []

            if city_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="city",
                        match=MatchValue(value=city_filter)
                    )
                )

            if project_type_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="project_type",
                        match=MatchValue(value=project_type_filter)
                    )
                )

            # Create filter if any conditions exist
            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )

            print(f"Filtered search returned: {len(search_results)} results")

            # If no filtered results, try without city filter (just project type)
            if len(search_results) == 0 and city_filter and project_type_filter:
                print("Trying search with just project_type filter...")
                project_only_filter = Filter(must=[
                    FieldCondition(
                        key="project_type",
                        match=MatchValue(value=project_type_filter)
                    )
                ])

                project_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=project_only_filter,
                    limit=limit,
                    with_payload=True
                )
                print(f"Project-type-only search returned: {len(project_results)} results")

                # If still no results, try without any filters
                if len(project_results) == 0:
                    print("Trying search without any filters...")
                    no_filter_results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=limit,
                        with_payload=True
                    )
                    print(f"No-filter search returned: {len(no_filter_results)} results")
                    search_results = no_filter_results[:10]  # Use some results at least
                else:
                    search_results = project_results

            print("=== END DEBUG ===")

            # Convert results to Document objects
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
        """
        Verify Qdrant connection and collection setup.
        """
        print("Setting up retriever - verifying Qdrant connection...")

        try:
            # Verify collection exists and is accessible
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Qdrant collection '{self.collection_name}' ready with {collection_info.vectors_count} vectors")

            # Test search functionality with a simple query
            test_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * len(self.embeddings.embed_query("test")),
                limit=1,
                with_payload=False
            )
            print("Qdrant search functionality verified")
            return True

        except Exception as e:
            print(f"Error setting up retriever: {str(e)}")
            return False

    async def process_comments(self, city_comments: List[ProjectCommentsResponse], append_mode: bool = True):
        """
        Complete pipeline to process multiple city comments and setup vector store.
        Args:
            city_comments: List of ProjectCommentsResponse objects
            append_mode: If True, append to existing vectorstore. If False, recreate.
        """
        all_documents = []
        for city_comment in city_comments:
            try:
                # Convert comments to Document objects
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
        """
        Extract structured city comments from PDF using Gemini AI
        """
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
                    print(f"Error cleaning up temp file {temp_path}: {str(e)}")
            return multiple_city_comments
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting comments from documents: {str(e)}")

    async def analyze_structural_design_with_gemini(self, design_file_path: str) -> DesignAnalysis:
            """
            Analyze a structural design document using Gemini AI to extract key information.
            Args:
                design_file_path: Path to the design document
            Returns:
                DesignAnalysis object containing extracted information
            """
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
                        return DesignAnalysis(filename=None, city=None, project_type=None, design_description=None, key_elements=[])
                    except Exception as e:
                        print(f"Error: Could not validate JSON against PlanCheckResponse model. {e}")
                        return DesignAnalysis(filename=None, city=None, project_type=None, design_description=None, key_elements=[])

            except Exception as e:
                print(f"Error cleaning up temp file {design_file_path}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error analyzing design document: {str(e)}")

    async def generate_structural_checklist(self, design_analysis: DesignAnalysis) -> ChecklistResponse:
        """
        Generate a contextual checklist based on design analysis and relevant past comments using Gemini,
        referencing a provided general checklist document.
        Args:
            design_analysis: Analysis of the uploaded design
        Returns:
            Generated checklist with categorized items
        """
        print("this is the print in generate list")
        relevant_comments = self.get_contextual_comments(design_analysis)
        comments_context = "\n".join([
            f"- {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''} (City: {doc.metadata.get('city', 'Unknown')}, Type: {doc.metadata.get('project_type', 'Unknown')})"
            for doc in relevant_comments[:30]
        ])
        # Todo: Fix prompt logic to fetch from text file
        # print('right after contextural comments', comments_context)
        # # Full content of the General Checklist.pdf provided in the prompt
        #
        # prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "generate_structural_checklist.txt")
        # with open(prompt_path, "r") as f:
        #     prompt_template = f.read()
        #
        # # Prepare the prompt variables
        # prompt = prompt_template.format(
        #     project_type=design_analysis.project_type,
        #     city=design_analysis.city,
        #     design_description=design_analysis.design_description,
        #     key_elements=", ".join(design_analysis.key_elements),
        #     num_comments=len(relevant_comments),
        #     comments_context=comments_context
        # )
        print('right before checklist llm')

        prompt= f"""You are an expert structural engineer. Based on the provided design analysis, past city review comments, and the attached General Structural QA Checklist, generate a comprehensive, detailed, and to-the-point checklist in valid JSON format.

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
        "city": "{design_analysis.city}",
        "project_type": "{design_analysis.project_type}",
        "design_focus": "Brief summary of main design focus areas"
      }},
      "checklist_items": [
        {{
          "category": "Structural",
          "item": "Verify foundation design calculations based on geotechnical report recommendations.",
          "description": "Cross-reference proposed foundation sizing, reinforcement, and bearing capacity calculations with the specific recommendations for this project's soil conditions (e.g., from section 3.1 Concrete Footing Design).",
          "priority": "High"
        }},
        {{
          "category": "Code Compliance",
          "item": "Review seismic design requirements for California Building Code (CBC) and ASCE 7.",
          "description": "Ensure seismic design parameters (SS, S1, SDC, Importance Factor) and lateral force resisting system (e.g., shear walls, per section 1.2 and 1.3) meet the latest California Building Code and ASCE 7 standards relevant to the project's city.",
          "priority": "High"
        }},
        {{
          "category": "Documentation",
          "item": "Ensure all structural drawings clearly differentiate existing, demolished, and new construction.",
          "description": "Verify drawing clarity and consistent use of symbols, especially for remodel/renovation projects as per section 5.1 and 5.3 of the General Checklist.",
          "priority": "Medium"
        }}
      ],
      "summary_of_key_concerns": "Summarize the overarching concerns derived from the relevant past city comments and their implications for the design.",
      "suggested_next_steps": [
        "Outline actionable next steps for the engineering team, prioritizing critical items based on the generated checklist and identified concerns.",
        "For example: 'Coordinate with geotechnical engineer for a revised report addressing liquefaction concerns.'"
      ]
    }}

    REQUIREMENTS:
    1.  **Select, adapt, and refine checklist items** from the "GENERAL STRUCTURAL QA CHECKLIST" provided, making them specifically relevant to the "DESIGN ANALYSIS" and any issues highlighted in "RELEVANT PAST CITY COMMENTS".
    2.  **Generate a comprehensive list of checklist items.** Do not limit the number of items; include all highly relevant and specific checks.
    3.  **Ensure items are "to the point" and actionable.** Avoid vague statements.
    4.  Categories for checklist items: Structural, Code Compliance, Documentation, Design Details, Safety.
    5.  Priority levels: High, Medium, Low (assign logically based on criticality and potential impact).
    6.  Provide a concise `summary_of_key_concerns` that synthesizes critical issues and common themes from the `RELEVANT PAST CITY COMMENTS`.
    7.  Outline clear and actionable `suggested_next_steps` for the project, directly addressing the checklist items and key concerns.
    8.  Return ONLY the JSON object - no additional text or formatting."""
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
                    return ChecklistResponse(filename=None, city=None, project_type=None, design_description=None, key_elements=[])
                except Exception as e:
                    print(f"Error: Could not validate JSON against PlanCheckResponse model. {e}")
                    return ChecklistResponse(filename=None, city=None, project_type=None, design_description=None, key_elements=[])

            return checklist_data

        except Exception as e:
            print(f"Error generating checklist with Gemini: {str(e)}")
            # return self._get_fallback_checklist(design_analysis, len(relevant_comments))

    def get_contextual_comments(self, design_analysis: DesignAnalysis, max_comments: int = 50,
                                context_comments: int = 30) -> List[Document]:
        """
        Retrieve contextually relevant comments based on design analysis using Qdrant.
        Args:
            design_analysis: Analysis of the uploaded design
            max_comments: Maximum number of comments to retrieve from Qdrant (default: 50)
            context_comments: Maximum number of comments to use in LLM context (default: 30)
        Returns:
            List of relevant comment documents
        """
        try:
            search_query = f"{design_analysis.design_description} {' '.join(design_analysis.key_elements)}"

            # Use actual city and project type from design analysis
            city_filter = design_analysis.city if design_analysis.city else None
            project_type_filter = design_analysis.project_type if design_analysis.project_type else None

            print(
                f"Searching for comments - City: {city_filter}, Project Type: {project_type_filter}, Max: {max_comments}")

            relevant_comments = self.search_similar_documents(
                query=search_query,
                limit=max_comments,
                city_filter=city_filter,
                project_type_filter=project_type_filter  # Added this line
            )

            print(
                f"Retrieved {len(relevant_comments)} relevant comments for city: {city_filter}, project type: {project_type_filter}")
            print(f"Will use top {min(context_comments, len(relevant_comments))} comments in LLM context")
            return relevant_comments

        except Exception as e:
            print(f"Error retrieving contextual comments: {str(e)}")
            return []

class MainService:
    def __init__(self):
        self.plan_check = PlanCheck()

        self.vectorstore_loaded = self.plan_check.load_vectorstore()
        if self.vectorstore_loaded:
            self.plan_check.setup_retriever()
            print("MainService initialized with existing vectorstore")
        else:
            print("MainService initialized - no existing vectorstore found")

    async def execute_plan_check(self, temp_file_paths: List[str]) -> PlanCheckResponse:
        """
        Execute plan check on the provided design set
        Args:
            temp_file_paths: List of paths to PDF files
        Returns:
            PlanCheckResponse Object
        """

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

            return plan_check_response

        except Exception as e:
            print(f"Error generating checklist with Gemini: {str(e)}")

    async def generate_structural_checklist(self, file_path: str) -> ChecklistResponse:
        """
        Complete pipeline to analyze a design and generate a contextual checklist.
        Args:
            file_path: Path to the structural design document
        Returns:
            GeneratedChecklist with items specific to the design
        """
        try:
            structural_design_analysis = await self.plan_check.analyze_structural_design_with_gemini(file_path)

            structural_checklist = await self.plan_check.generate_structural_checklist(structural_design_analysis)
            return structural_checklist
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating design checklist: {str(e)}")

    async def ingest_comments(self, temp_file_paths: List[str]) -> Dict:
        """
        Ingest comments from PDF files into Qdrant vector store.
        Args:
            temp_file_paths: List of paths to PDF files
        Returns:
            Dictionary with ingestion status
        """
        try:
            # Extract comments from PDFs
            project_comments_data = await self.plan_check.extract_comments_with_gemini(temp_file_paths)

            # Process and store in vector store
            await self.plan_check.process_comments(project_comments_data, append_mode=True)

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