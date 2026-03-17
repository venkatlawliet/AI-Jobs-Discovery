from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uvicorn

from data_loader import JobsDataLoader
from search_engine import JobSearchEngine

app = FastAPI(title="AI Search API")

# Global Engine Instance
engine = None

class SearchRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

@app.on_event("startup")
async def startup_event():
    global engine
    print("Loading Dataset into Memory...")
    max_rows = int(os.getenv("MAX_ROWS", 10000))
    loader = JobsDataLoader(max_rows=max_rows)
    meta, me, mi, mc = loader.load()
    engine = JobSearchEngine(meta, me, mi, mc)
    print("Engine Ready!")

@app.post("/api/search")
async def search(req: SearchRequest):
    if not engine:
        return {"error": "Engine not loaded."}
        
    parsed_intent = engine.parse_query(req.query, req.history)
    results = engine.search(parsed_intent, top_k=10)
    
    # Update History with proper JSON serialization
    new_history = req.history.copy()
    new_history.append({"role": "user", "content": req.query})
    new_history.append({"role": "assistant", "content": f"Understood intent: {json.dumps(parsed_intent)}"})
    
    cost = (engine.tracker.prompt_tokens / 1_000_000) * 0.15 + (engine.tracker.completion_tokens / 1_000_000) * 0.60 + (engine.tracker.embedding_tokens / 1_000_000) * 0.02
    total_tokens = engine.tracker.prompt_tokens + engine.tracker.completion_tokens + engine.tracker.embedding_tokens
    
    return {
        "intent": parsed_intent,
        "results": results,
        "history": new_history,
        "metrics": {
            "total_tokens": total_tokens,
            "estimated_cost": cost
        }
    }

# Mount static files for the HTML frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
