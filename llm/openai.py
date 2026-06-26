import os
import json
import ast
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIResponse:
    def __init__(self, text: str):
        self.text = text


class OpenAIModel:
    def __init__(self, model_name: str = "gpt-5-mini", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_content(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        max_completion_tokens: int = 2048
    ):
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_completion_tokens
        )

        message = response.choices[0].message
        if isinstance(message, dict):
            text = message.get("content", "")
        else:
            text = getattr(message, "content", None)
            if text is None:
                text = getattr(message, "text", None)
            if text is None:
                try:
                    text = message["content"]
                except Exception:
                    text = str(message)

        return OpenAIResponse(text.strip() if isinstance(text, str) else "")


def get_openai_client():
    """
    Initialize and configure OpenAI client for GPT-5 Mini.

    Returns:
        Configured OpenAI model wrapper.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return OpenAIModel(model_name=model_name, api_key=api_key)


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model=None
) -> BaseModel:
    """
    Generate structured output from OpenAI.

    Args:
        prompt: Input prompt.
        response_model: Pydantic response model for validation.
        model: Optional OpenAI model wrapper instance.

    Returns:
        Parsed response model instance.
    """
    if model is None:
        model = get_openai_client()

    system_message = {
        "role": "system",
        "content": (
            "You are a precise financial analyst. Extract the requested financial metrics from the provided context. "
            "Return only valid JSON with the requested fields. If a field is missing, return null for that field."
        )
    }

    messages = [
        system_message,
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = model.generate_content(
        messages=messages,
        max_completion_tokens=2048
    )

    text = response.text
    print("[debug] OpenAI raw text response:\n", text)

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
        if isinstance(json_text, str) and json_text.strip() in ("null", "None", ""):
            if hasattr(response_model, "model_construct"):
                return response_model.model_construct()
            try:
                return response_model()
            except Exception:
                return response_model.model_validate({})

        parsed_dict = json.loads(json_text)
        return response_model.model_validate(parsed_dict)

    except json.JSONDecodeError as e:
        print(f"[debug] JSON decode error: {e}\njson_text:\n{json_text}\nraw text:\n{text}")
        try:
            parsed = ast.literal_eval(json_text)
            if isinstance(parsed, dict):
                return response_model.model_validate(parsed)
        except Exception:
            pass

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
