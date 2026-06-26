import os
import json
import re
import ast
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()


def get_gemini_client():
    """
    Initialize and configure Gemini client.

    Returns:
        Configured Gemini model ready for use.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model=None
) -> BaseModel:
    """
    Generate structured output from Gemini.

    Args:
        prompt: Input prompt.
        response_model: Pydantic response model for validation.
        model: Gemini model instance (optional, defaults to gemini-2.5-flash).

    Returns:
        Parsed response model instance.
    """
    if model is None:
        model = get_gemini_client()

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Generate response using Gemini
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048
        )
    )

    text = response.text
    print("[debug] Gemini raw text response:\n", text)

    # Try to extract JSON from the response using a brace-matching extractor
    def _extract_json_like(s: str) -> str:
        if not isinstance(s, str) or not s:
            return ""
        start = s.find("{")
        if start == -1:
            return s
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return s[start:]

    json_text = _extract_json_like(text)

    try:
        # Handle explicit 'null' responses
        if isinstance(json_text, str) and json_text.strip() in ("null", "None", ""):
            if hasattr(response_model, "model_construct"):
                return response_model.model_construct()
            try:
                return response_model()
            except Exception:
                return response_model.model_validate({})

        # Parse JSON and validate with Pydantic
        parsed_dict = json.loads(json_text)
        return response_model.model_validate(parsed_dict)

    except json.JSONDecodeError as e:
        print(f"[debug] JSON decode error: {e}\njson_text:\n{json_text}\nraw text:\n{text}")
        # Fallback: try to parse Python-style dicts using ast.literal_eval
        try:
            parsed = ast.literal_eval(json_text)
            if isinstance(parsed, dict):
                return response_model.model_validate(parsed)
        except Exception:
            pass

        # Try to construct empty model
        if hasattr(response_model, "model_construct"):
            return response_model.model_construct()
        return response_model.model_validate({})


class SentenceTransformerEmbeddings:
    """
    Wrapper for SentenceTransformer to be compatible with LangChain's embeddings interface.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        """Embed a single query."""
        return self.model.encode(text, convert_to_tensor=False).tolist()


def get_embedding_model():
    """
    Get embedding model for semantic similarity.
    
    Uses sentence-transformers for local embeddings with LangChain compatibility.
    """
    try:
        return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    except ImportError:
        raise ImportError("sentence-transformers is required for embeddings. Install with: pip install sentence-transformers")
