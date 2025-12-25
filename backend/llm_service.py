import os
import google.generativeai as genai
import pandas as pd
import json
from fastapi import UploadFile

# Configure Gemini
# You should ensure GOOGLE_API_KEY is set in your .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-2.0-flash')

class LLMService:
    async def process_file_query(self, file: UploadFile | None, query: str):
        try:
            # Default prompt for file-only analysis
            if not query or query.strip() == "":
                if file:
                    query = "Analyze the provided data and identify the most significant location or the first valid coordinate pair found."
                else:
                    return {"error": "Please provide a query or upload a file."}

            data_preview = "No file uploaded. Use your general knowledge."
            columns = "None"
            
            # 1. Read File (if provided)
            if file:
                filename = file.filename
                content = await file.read()
                
                # Save temp file
                temp_path = f"server_upload_{filename}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                    
                # 2. Parse Data
                df = None
                if filename.endswith(".csv"):
                    df = pd.read_csv(temp_path)
                elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                    df = pd.read_excel(temp_path)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                if df is not None:
                    data_preview = df.head(5).to_string()
                    columns = df.columns.to_list()

            # 3. Construct Prompt
            prompt = f"""
            You are a Geographic Data Analyst.
            
            DATA CONTEXT:
            Cols: {columns}
            Preview:
            {data_preview}
            
            USER QUERY: "{query}"
            
            INSTRUCTIONS:
            1. Identified the location the user is interested in.
            2. If a file is provided (Data Context is not empty), look for the location in the data rows.
            3. If NO file is provided, use your internal knowledge to find the coordinates of the place mentioned in the query.
            4. Return the result strictly as a valid JSON object.
            
            JSON Schema:
            {{
                "lat": float,
                "lon": float,
                "location_name": "string (name of the identified place)",
                "explanation": "string (brief explanation)"
            }}
            
            Do not include any markdown formatting (like ```json), just the raw JSON string.
            """
            
            # 4. Call Gemini
            response = model.generate_content(prompt)
            result_text = response.text
            
            # Cleanup markdown if present
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            result_json = json.loads(result_text)
            
            return result_json

        except Exception as e:
            print(f"LLM Error: {e}")
            return {"error": str(e)}

llm_service = LLMService()
