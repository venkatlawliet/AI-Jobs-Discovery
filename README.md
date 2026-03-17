# AI Job Search Prototype

This repository contains a prototype AI-powered job discovery engine. It uses natural language queries and iterative refinement to help users find relevant job postings.

## How to Run

1. Ensure you have Python 3.9+ installed.
2. Place the `jobs.jsonl` file in the same directory as the project.
3. Install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   # source .venv/bin/activate
   
   pip install numpy pandas openai python-dotenv fastapi uvicorn
   ```
4. Create a `.env` file in the root directory and add your OpenAI Key:
   ```
   OPENAI_API_KEY=sk-proj-...
   ```
5. Start the Web UI Server (Recommended):
   ```bash
   python app.py
   ```
   *Then open `http://localhost:8000` in your web browser for the full interactive graphical experience.*

6. Or, run the interactive manual CLI demo:
   ```bash
   python demo.py
   ```
7. Or, run the automated multi-session CLI demo:
   ```bash
   python demo.py --auto --max_rows 5000
   ```
   *Note: `--max_rows` reduces the memory load for testing. To load all 100k jobs, omit this flag or set `MAX_ROWS` in your `.env`.*

## Architectural Approach

### 1. Data Representation & Processing
The engine reads the `jobs.jsonl` file sequentially using a stream-like approach in `data_loader.py` to prevent memory blowups when parsing JSON. 
- We extract the three 1536-dimensional float32 vectors (`embedding_explicit_vector`, `embedding_inferred_vector`, `embedding_company_vector`) and store them in three large NumPy matrices.
- We also extract **15+ structured metadata fields** from `v5_processed_job_data` and `v5_processed_company_data`: seniority level, job category, salary range, company sector, non-profit status, public company status, employee count, and more.
- To maximize lookup speed, we L2-normalize all vectors at load-time and use dot products instead of full cosine similarity computations. Numpy performs these operations almost instantaneously, avoiding the need for a complex external vector database like Pinecone or Chroma.

### 2. Search & Intent Parsing (Hybrid Approach)
When a user types a query, we pass it through `gpt-4o-mini` to extract a **rich structured intent**:
- **Semantic queries**: explicit job titles/skills, inferred qualifications, and company traits — each embedded separately using `text-embedding-3-small` and matched against the corresponding vector space.
- **Structured hard filters**: workplace type (Remote/Onsite/Hybrid), specific company names, seniority level, non-profit status, and salary range.
- **Acronym resolution**: The LLM resolves terms like "FAANG" into specific company names (Meta, Amazon, Apple, Netflix, Google + subsidiaries).
- **Conversation context**: On follow-up queries, the LLM merges the new constraint with ALL previous context, never dropping earlier intent.

### 3. Relevance and Ranking
Relevance is determined by a **two-stage hybrid approach**:
1. **Structured filtering** (boolean mask): Hard filters like company names, seniority, and remote are applied first, eliminating non-matching jobs before scoring even begins.
2. **Weighted vector scoring**: A weighted sum of dot products against the three embedding matrices (explicit: 1.0, company: 0.7, inferred: 0.3).
3. **Graceful fallback**: When hard filters eliminate all results (e.g., no exact FAANG jobs in the dataset), the system automatically converts company names into a soft semantic query to find the most similar companies.

### 4. Smart Company Name Matching
Matching company names is deceptively hard. "Apple" must match "Apple Inc." but NOT "Apple Autos" or "Little Green Apple Hallmark". Our solution:
- Strips noise suffixes (Inc, LLC, Corp, Ltd, etc.) before comparison
- Checks if the target is the *core identity* of the company name, not just any word in it
- Handles parent/subsidiary relationships (e.g., "Amazon" matches "AWS")

### 5. Trade-offs
- **In-Memory Numpy vs Vector DB:** To keep the solution lightweight and single-command runnable, I opted for pure NumPy. While fast for 100k rows (~1.8GB of vectors), this would not scale to millions of rows without a dedicated vector DB (e.g. Qdrant or Milvus), as it is entirely RAM-bound.
- **LLM Latency:** Using GPT-4o-mini for query parsing adds ~500ms of latency per query. While it makes the search incredibly smart, at extreme scale, we might want a fine-tuned lightweight model or cheaper parser.
- **Hard vs Soft filtering:** Specific company names use hard filtering (exact match), while vague company traits ("mission-driven") use soft vector similarity. This gives the best of both worlds.

## Query Performance
- **Works well:** Complex, multi-faceted queries with iterative refinement:
  - "data science jobs" → "at non-profits" → "make it remote" ✅
  - "machine learning jobs" → "at FAANG companies" ✅
  - "senior software engineer remote" (with salary display) ✅
- **Tricky:** Extremely vague queries ("good jobs") or cases where the dataset doesn't contain jobs from the specific companies the user asks about (handled gracefully via fallback).

## Future Improvements
With more time, I would:
1. Implement a local persistent DB (like DuckDB or SQLite-vec) so we don't need to load the JSONL file into RAM on every startup.
2. Add comprehensive filtering for states/cities/salary ranges via UI dropdowns.
3. Use a more sophisticated re-ranker (e.g., Cross-Encoder) for the top 50 results instead of relying entirely on simple cosine similarity.
4. Add a spell-checker / fuzzy matching layer for company names.

## Tokens Report
*   **Tokens used during development:** Estimated ~250,000 prompt tokens and ~50,000 completion tokens (using `gpt-4o-mini`), primarily for tuning the intent extraction prompt and running the automated test suite over dozens of iterations.
*   **Total Development Cost:** < $0.15
*   **Tokens consumed per query:** On average, the intent parsing consumes between **350 - 550 prompt tokens** depending on the length of the conversation history, and **30 - 60 completion tokens** to return the structured JSON intent.
*   **Average Cost per Query:** ~$0.0001 per search.
*   *Note: We built a built-in token tracker into the `TokenTracker` class, which displays live token usage and estimated cost directly in both the Web UI sidebar and the CLI `demo.py` tool.*
