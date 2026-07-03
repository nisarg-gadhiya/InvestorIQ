import os
import re
from types import SimpleNamespace
import numpy as np

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm.openai import get_structured_completion
from vectorstore.chromadb_store import ChromaDBVectorStore

load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


class Retriever:
    def __init__(self, client):
        self.client = client

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """
        Retrieve relevant chunks from ChromaDB.
        """
        where_filter = None
        if company and year:
            where_filter = {
                "$and": [
                    {"company": {"$eq": company}},
                    {"year": {"$eq": str(year)}}
                ]
            }
        elif company:
            where_filter = {"company": {"$eq": company}}
        elif year:
            where_filter = {"year": {"$eq": str(year)}}

        results = self.client.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "embeddings"]
        )

        documents = []
        if results and results.get("documents"):
            texts = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            embeddings = results["embeddings"][0] if results.get("embeddings") else []

            for i, text in enumerate(texts):
                metadata = metadatas[i] if i < len(metadatas) else {}
                embedding = embeddings[i] if i < len(embeddings) else None
                documents.append(
                    SimpleNamespace(
                        page_content=text,
                        embedding=embedding,
                        page=metadata.get("page"),
                        company=metadata.get("company"),
                        source_file=metadata.get("source_file")
                    )
                )

        return documents


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int
) -> str:
    """
    Retrieve broad financial context from the vector store.
    Uses multiple targeted queries to get comprehensive context.
    """
    # Query 1: Financial statements
    financial_query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    revenue, net income, assets, liabilities
    """
    
    # Query 2: Risks
    risks_query = f"""
    Risk factors, challenges, uncertainties,
    threats, market risks, operational risks
    """
    
    # Query 3: Growth and opportunities
    growth_query = f"""
    Growth drivers, opportunities, strategic initiatives,
    new products, expansion, market growth
    """

    # Query 4: Company and year-specific financial results
    company_year_query = f"""
    {company} {year} annual report, financial results,
    revenue, net income, operating income, cash flow, assets, liabilities
    """

    all_docs = []
    
    # Retrieve from each query
    for query in [financial_query, risks_query, growth_query, company_year_query]:
        documents = retriever.invoke(
            query=query,
            company=company,
            year=year,
            top_k=20
        )
        all_docs.extend(documents)

    SIMILARITY_THRESHOLD = 0.92  # tunable — above this = treat as duplicate

    unique_docs = []
    unique_embeddings = []

    for doc in all_docs:
        embedding = doc.embedding  # returned by ChromaDB alongside text

        # Check against every already-kept chunk's embedding
        is_duplicate = False
        for kept_embedding in unique_embeddings:
            similarity = cosine_similarity(embedding, kept_embedding)
            if similarity > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(doc)
            unique_embeddings.append(embedding)
    
    return "\n\n".join(
        doc.page_content
        for doc in unique_docs[:30]  # Limit to top 30 to avoid token issues
    )

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm > 0 else 0.0


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt with detailed instructions.
    """
    return f"""
You are an expert financial analyst specializing in corporate filings.

Company: {company}
Year: {year}

Context from Annual Report:
{context}

Extract the following information from the context:

1. **Revenue**: Total annual revenue. Search for values such as "revenue", "net sales", "total revenues", "sales revenue", or table entries.
2. **Net Income**: Bottom line profit. Search for "net income", "net earnings", "profit", "loss".
3. **Operating Income**: Operating profit. Search for "operating income", "operating profit", "income from operations".
4. **Cash Flow from Operating Activities**: Operating cash flow. Search for "cash flow from operating activities", "operating cash flow".
5. **Total Assets**: Total company assets. Search for "total assets", "assets".
6. **Total Liabilities**: Total liabilities. Search for "total liabilities", "liabilities".
7. **Top Risk Factors**: List 3-5 main risks or challenges mentioned. Search for: "risks", "challenges", "uncertainties", "threats", "headwinds".
8. **Top Growth Drivers**: List 3-5 main growth opportunities or drivers. Search for: "growth", "opportunities", "strategic initiatives", "new products", "expansion", "innovation".

**IMPORTANT INSTRUCTIONS — READ CAREFULLY:**
- Extract ONLY from the text provided in the Context section above.
- Do NOT use your training knowledge, memory, or any information not explicitly present in the Context.
- If you are not certain a value appears in the Context, return null for that field.
- Do NOT infer, estimate, or calculate values not directly stated.
- Do NOT round or approximate — return the exact figure as written in the Context.
- For Risk Factors and Growth Drivers only: you may infer from business discussion even if not explicitly labeled.
- For all numeric fields: if the report states values are "in thousands" or "in millions", convert to full dollars.
- Return null for any numeric field where you are less than fully certain the value is present in the Context.
- Return VALID JSON ONLY. No markdown, no explanation, no additional keys.

Example output format:
{{
  "Revenue": "$1,000,000",
  "Net Income": "$200,000",
  "Operating Income": "$150,000",
  "Cash Flow from Operating Activities": "$120,000",
  "Total Assets": "$5,000,000",
  "Total Liabilities": "$2,500,000",
  "Top Risk Factors": ["Factor 1", "Factor 2"],
  "Top Growth Drivers": ["Driver 1", "Driver 2"]
}}

Do not include any additional keys, markdown, or explanations. If a field cannot be found despite a thorough search, use null for that field.
"""


def normalize_currency(value: str | int | None) -> str | None:
    """Normalize currency values to a consistent string format."""
    if value is None:
        return None

    if isinstance(value, int):
        return f"${value:,}"

    text = str(value).strip()
    if text.lower() == "null":
        return None

    # Remove parentheses for negative values and normalize sign
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()

    # Normalize currency symbol and commas
    cleaned = re.sub(r"[^0-9\.\-]", "", text)
    if cleaned == "":
        return text

    try:
        if "." in cleaned:
            number = float(cleaned)
            integer = int(round(number))
            formatted = f"${integer:,}"
        else:
            number = int(cleaned)
            formatted = f"${number:,}"
    except ValueError:
        return text

    if is_negative:
        formatted = f"({formatted[1:]})"

    return formatted


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract KPIs using RAG.
    """
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year
    )

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    metrics = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )

    result = metrics.model_dump()

    # Normalize financial KPI formats before saving/display
    for key in [
        "Revenue",
        "Net Income",
        "Operating Income",
        "Cash Flow from Operating Activities",
        "Total Assets",
        "Total Liabilities"
    ]:
        if key in result:
            result[key] = normalize_currency(result[key])

    return result


def main() -> None:
    company = "Apple"
    year = 2024

    vector_store = ChromaDBVectorStore(
        db_path=os.getenv("CHROMA_DB_PATH", "./chroma_data")
    )

    retriever = Retriever(
        vector_store.collection
    )

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )

    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)


    from database.save_metrics import save_metrics

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )

if __name__ == "__main__":
    main()