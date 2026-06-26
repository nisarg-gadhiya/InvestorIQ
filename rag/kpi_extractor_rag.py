import os
from types import SimpleNamespace

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, Field

from llm.gemini import get_structured_completion
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
        filter_expr = None

        if company and year:
            filter_expr = (
                f"company eq '{company}' "
                f"and year eq '{year}'"
            )

        results = (
            self.client.search(
                search_text=query,
                top=top_k,
                filter=filter_expr
            )
            if filter_expr
            else self.client.search(
                search_text=query,
                top=top_k
            )
        )

        documents = []

        for result in results:
            content = result.get("content", "")
            documents.append(
                SimpleNamespace(
                    page_content=content
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

    all_docs = []
    
    # Retrieve from each query
    for query in [financial_query, risks_query, growth_query]:
        documents = retriever.invoke(
            query=query,
            company=company,
            year=year,
            top_k=10
        )
        all_docs.extend(documents)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    return "\n\n".join(
        doc.page_content
        for doc in unique_docs[:30]  # Limit to top 30 to avoid token issues
    )


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

1. **Revenue**: Total annual revenue (look for "revenue", "net sales", "total revenues")
2. **Net Income**: Bottom line profit (look for "net income", "net earnings")
3. **Operating Income**: Operating profit (look for "operating income", "operating profit")
4. **Cash Flow from Operating Activities**: Operating cash flow
5. **Total Assets**: Total company assets
6. **Total Liabilities**: Total liabilities
7. **Top Risk Factors**: List 3-5 main risks or challenges mentioned. Look for: "risks", "challenges", "uncertainties", "threats". If multiple mentions, pick the most significant ones. Return as a list.
8. **Top Growth Drivers**: List 3-5 main growth opportunities or drivers. Look for: "growth", "opportunities", "strategic initiatives", "new products", "expansion". Return as a list.

**IMPORTANT INSTRUCTIONS:**
- Extract ONLY from the provided context
- Return null ONLY if the information is genuinely not in the context
- For Risk Factors and Growth Drivers: Even if not labeled as such, infer them from the business discussion
- Financial values must match the report exactly (include currency symbols and formatting)
- For lists: Return as an array of strings
- Return VALID JSON ONLY, no markdown formatting

If any field cannot be found despite thorough search, return null for that field.
"""


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

    return metrics.model_dump()


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