import json
import os
from typing import List

# Core libraries
from pydantic import BaseModel

# Gemini:
from google import genai
from google.genai import types

# Load Environment variable
from dotenv import load_dotenv
load_dotenv()

# Load Gemini Client:
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

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

class PlanCheck:
    def __init__(self):
        self.gemini_client = client

class MainService:
    def __init__(self):

        self.plan_check = PlanCheck()

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
