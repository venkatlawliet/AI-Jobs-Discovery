import os
import json
import re
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0

    def add_llm_tokens(self, usage):
        if usage:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens

    def add_embedding_tokens(self, usage):
        if usage:
            self.embedding_tokens += usage.prompt_tokens

    def report(self):
        cost = (self.prompt_tokens / 1_000_000) * 0.15 + \
               (self.completion_tokens / 1_000_000) * 0.60 + \
               (self.embedding_tokens / 1_000_000) * 0.02
        return f"Tokens Used - LLM In: {self.prompt_tokens}, LLM Out: {self.completion_tokens}, Emb In: {self.embedding_tokens}. Approx Cost: ${cost:.6f}"


class JobSearchEngine:
    """
    Hybrid search engine combining:
    1. LLM-powered intent parsing (extracts structured filters + semantic queries)
    2. Vector similarity search (cosine similarity across 3 embedding spaces)
    3. Structured filtering (company name matching, seniority, remote, salary, etc.)
    """
    
    SYSTEM_PROMPT = """You are a job search intent parser. Given a user's job search query (and possibly previous conversation context), extract a COMPLETE, MERGED search intent as a JSON object.

CRITICAL RULES:
- On follow-up queries, you must MERGE the new constraint with ALL previous constraints. Never drop previous context.
- If the user mentions FAANG or MAANG, resolve to these company names: ["Meta", "Facebook", "Amazon", "AWS", "Apple", "Netflix", "Google", "Alphabet", "YouTube", "Instagram", "WhatsApp"]
- If the user mentions Big Tech, also include: ["Microsoft", "Meta", "Amazon", "Apple", "Google", "Alphabet"]
- If the user mentions specific companies by name or acronym, list them in company_names.
- Distinguish between SPECIFIC company filters (company_names) vs GENERAL company traits (company_query).
- For company_names, include all common brand names, parent companies, and subsidiaries to maximize matching.

Output this exact JSON structure:
{
  "explicit_query": "Job titles and specific skills the user wants. E.g. 'Machine Learning Engineer, Data Scientist, ML, Python, TensorFlow'",
  "inferred_query": "Related/implied qualifications and skills. E.g. 'Deep Learning, Neural Networks, Statistics, PyTorch'",
  "company_query": "General company characteristics (NOT specific names). E.g. 'mission-driven non-profit', 'healthcare startup', 'large tech company'. Empty string if not mentioned.",
  "filters": {
    "workplace_type": "Remote | Onsite | Hybrid | null",
    "company_names": ["list", "of", "specific", "company", "names", "to", "match"],
    "seniority": "Entry Level | Mid Level | Senior Level | Lead | null",
    "location": "State name or city name if the user specifies a location. E.g. 'Maryland', 'New York', 'San Francisco'. null if not mentioned.",
    "is_non_profit": true/false/null,
    "min_salary": null or number,
    "max_salary": null or number
  }
}

Use null for any filter not mentioned. Keep queries concise but complete."""

    def __init__(self, metadata, mat_explicit, mat_inferred, mat_company, token_tracker=None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.metadata = metadata
        
        # Pre-normalize matrices for dot-product cosine similarity
        self.mat_explicit = self._normalize(mat_explicit)
        self.mat_inferred = self._normalize(mat_inferred)
        self.mat_company = self._normalize(mat_company)
        
        self.tracker = token_tracker or TokenTracker()
        
        # Pre-build company name index for fast substring search
        self._company_names_lower = [m.get('company_lower', '') for m in metadata]

    def _normalize(self, mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms

    def _embed(self, texts):
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        self.tracker.add_embedding_tokens(response.usage)
        return [np.array(res.embedding, dtype=np.float32) for res in response.data]

    def parse_query(self, user_query, conversation_history=None):
        """Use LLM to extract structured intent from natural language query."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        if conversation_history:
            messages.extend(conversation_history)
            messages.append({
                "role": "user",
                "content": f"The user's new message is: \"{user_query}\"\n\nMerge this with all previous context and output the COMPLETE updated JSON intent."
            })
        else:
            messages.append({"role": "user", "content": user_query})

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistent structured output
        )
        self.tracker.add_llm_tokens(response.usage)
        
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            # Normalize filters
            if 'filters' not in parsed:
                parsed['filters'] = {}
            return parsed
        except json.JSONDecodeError:
            return {
                "explicit_query": user_query,
                "inferred_query": "",
                "company_query": "",
                "filters": {}
            }

    @staticmethod
    def _company_match(job_company_lower, target_names):
        """
        Smart company name matching that avoids false positives.
        
        Strategy:
        - Split both names into tokens
        - The first token of the job company must match a target name or its first token
        - OR the job company starts with the target name
        - This prevents 'Apple Autos' from matching 'Apple' (the tech company)
          because 'Apple Autos' has a second word that makes it a different company.
        - But 'Apple' matches 'Apple', 'Apple Inc', etc.
        
        For multi-word targets like 'Amazon Web Services', we check if the job
        company starts with the target.
        """
        # Suffixes that don't change the company identity
        NOISE_SUFFIXES = {'inc', 'inc.', 'llc', 'ltd', 'corp', 'co', 'company', 
                          'corporation', 'group', 'holdings', 'industries', 'plc',
                          'technologies', 'technology', 'labs', 'studios'}
        
        job_tokens = job_company_lower.split()
        # Strip noise suffixes from job company for comparison
        job_core = [t for t in job_tokens if t.rstrip('.,') not in NOISE_SUFFIXES]
        if not job_core:
            job_core = job_tokens
        
        for target in target_names:
            target_tokens = target.split()
            
            # Strategy 1: Exact match (after stripping noise)
            target_core = [t for t in target_tokens if t.rstrip('.,') not in NOISE_SUFFIXES]
            if not target_core:
                target_core = target_tokens
            
            if job_core == target_core:
                return True
            
            # Strategy 2: Job company starts with the target
            if job_company_lower.startswith(target + ' ') or job_company_lower == target:
                # But exclude if the next word makes it a different company
                # e.g., "apple autos" starts with "apple" but is not Apple Inc
                if len(job_tokens) > len(target_tokens):
                    remaining = job_tokens[len(target_tokens):]
                    remaining_stripped = [t.rstrip('.,') for t in remaining]
                    # If all remaining words are noise suffixes, it's a match
                    if all(w in NOISE_SUFFIXES for w in remaining_stripped):
                        return True
                    # Otherwise it's a different company
                else:
                    return True
            
            # Strategy 3: Target starts with job company (e.g., job="amazon", target="amazon web services")
            if target.startswith(job_company_lower + ' ') or target == job_company_lower:
                return True
            
            # Strategy 4: For single-word targets, check if first core token matches
            if len(target_core) == 1 and len(job_core) == 1:
                if job_core[0] == target_core[0]:
                    return True
        
        return False

    def _build_filter_mask(self, parsed_intent):
        """
        Build a boolean mask array. True = job passes all hard filters.
        This is applied BEFORE vector scoring so filtered jobs are excluded.
        """
        num_jobs = len(self.metadata)
        mask = np.ones(num_jobs, dtype=bool)  # Start: all jobs pass
        
        filters = parsed_intent.get('filters', {})
        
        # --- Workplace Type Filter ---
        wp = filters.get('workplace_type')
        if wp and wp.lower() != 'null':
            wp_lower = wp.lower().strip()
            for i, meta in enumerate(self.metadata):
                if meta.get('workplace_type', '').lower() != wp_lower:
                    mask[i] = False
        
        # --- Company Name Filter (smart matching, case-insensitive) ---
        company_names = filters.get('company_names', [])
        if company_names and company_names != [None]:
            company_names = [n.lower().strip() for n in company_names if n]
            if company_names:
                for i in range(num_jobs):
                    job_company = self._company_names_lower[i]
                    matched = self._company_match(job_company, company_names)
                    if not matched:
                        mask[i] = False
        
        # --- Seniority Filter ---
        seniority = filters.get('seniority')
        if seniority and seniority.lower() not in ('null', 'none', ''):
            sen_lower = seniority.lower().strip()
            for i, meta in enumerate(self.metadata):
                job_sen = meta.get('seniority', '').lower()
                if sen_lower not in job_sen and job_sen not in sen_lower:
                    mask[i] = False
        
        # --- Non-Profit Filter ---
        is_non_profit = filters.get('is_non_profit')
        if is_non_profit is True:
            for i, meta in enumerate(self.metadata):
                if not meta.get('is_non_profit', False):
                    mask[i] = False
        
        # --- Salary Filter ---
        min_sal = filters.get('min_salary')
        max_sal = filters.get('max_salary')
        if min_sal and isinstance(min_sal, (int, float)):
            for i, meta in enumerate(self.metadata):
                job_max = meta.get('salary_max')
                if job_max and job_max < min_sal:
                    mask[i] = False
        if max_sal and isinstance(max_sal, (int, float)):
            for i, meta in enumerate(self.metadata):
                job_min = meta.get('salary_min')
                if job_min and job_min > max_sal:
                    mask[i] = False
        
        # --- Location Filter (state or city substring match) ---
        location = filters.get('location')
        if location and location.lower() not in ('null', 'none', ''):
            loc_lower = location.lower().strip()
            for i, meta in enumerate(self.metadata):
                # Check against structured state list first
                states = meta.get('states_lower', [])
                loc_str = meta.get('location_lower', '')
                matched = False
                # Check if location matches any state
                for state in states:
                    if loc_lower in state or state in loc_lower:
                        matched = True
                        break
                # Also check location string (city, state)
                if not matched and loc_lower in loc_str:
                    matched = True
                if not matched:
                    mask[i] = False
        
        return mask

    def search(self, parsed_intent, top_k=10):
        """
        Hybrid search: structured filtering + weighted vector similarity.
        When hard company filters match 0 jobs, falls back to soft vector matching
        by injecting the company names into the company_query for embedding search.
        """
        # Step 1: Build filter mask
        mask = self._build_filter_mask(parsed_intent)
        num_passing = mask.sum()
        
        # Make a mutable copy of the intent for potential fallback adjustments
        working_intent = dict(parsed_intent)
        
        if num_passing == 0:
            # Hard filters eliminated everything.
            # ONLY fallback if there was a company names filter.
            # We never blindly drop other hard filters (like location, seniority, remote)
            # because if a user asks for Maryland or NYC, we shouldn't show them Chicago.
            filters = working_intent.get('filters', {})
            company_names = filters.get('company_names', [])
            if company_names:
                # Inject the company names as a semantic company_query
                existing_cq = working_intent.get('company_query', '')
                names_str = ', '.join([n for n in company_names if n])
                if existing_cq:
                    working_intent['company_query'] = f"{existing_cq}, {names_str}"
                else:
                    working_intent['company_query'] = f"Companies like {names_str}"
                # Remove the hard filter so we get soft matches
                filters['company_names'] = []
            
                # Rebuild mask WITHOUT the company name filter, but KEEP all other filters
                mask = self._build_filter_mask(working_intent)
                num_passing = mask.sum()
        
        # Step 2: Embed semantic queries
        queries_to_embed = []
        keys = []
        
        explicit_q = working_intent.get('explicit_query', '')
        inferred_q = working_intent.get('inferred_query', '')
        company_q = working_intent.get('company_query', '')
        
        if explicit_q:
            queries_to_embed.append(explicit_q)
            keys.append('explicit')
        if inferred_q:
            queries_to_embed.append(inferred_q)
            keys.append('inferred')
        if company_q:
            queries_to_embed.append(company_q)
            keys.append('company')
            
        if not queries_to_embed:
            return []
            
        embeddings = self._embed(queries_to_embed)
        emb_dict = dict(zip(keys, embeddings))
        
        # Step 3: Compute weighted similarity scores
        num_jobs = len(self.metadata)
        total_scores = np.zeros(num_jobs, dtype=np.float32)
        
        # Weights: explicit is most important, company traits next, inferred is supplemental
        weights = {'explicit': 1.0, 'company': 0.7, 'inferred': 0.3}
        weight_sum = 0.0
        
        for key in ['explicit', 'inferred', 'company']:
            if key in emb_dict:
                vec = emb_dict[key]
                vec = vec / (np.linalg.norm(vec) or 1)
                mat = getattr(self, f'mat_{key}')
                scores = mat.dot(vec)
                w = weights[key]
                total_scores += scores * w
                weight_sum += w
        
        if weight_sum > 0:
            total_scores /= weight_sum
        
        # Step 4: Apply filter mask (zero out filtered jobs)
        total_scores[~mask] = -1.0
        
        # Step 5: Get top-k
        top_indices = np.argsort(total_scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if total_scores[i] > 0:
                job = self.metadata[i].copy()
                job['score'] = float(total_scores[i])
                # Remove internal fields from response
                job.pop('company_lower', None)
                job.pop('company_industries', None)
                results.append(job)
                
        return results

if __name__ == '__main__':
    from data_loader import JobsDataLoader
    loader = JobsDataLoader(max_rows=500)
    meta, me, mi, mc = loader.load()
    engine = JobSearchEngine(meta, me, mi, mc)
    
    # Test 1: Simple search
    print("=== Test 1: machine learning jobs ===")
    intent = engine.parse_query("machine learning jobs")
    print("Intent:", json.dumps(intent, indent=2))
    results = engine.search(intent, top_k=3)
    for r in results:
        print(f"  {r['title']} @ {r['company']} ({r['workplace_type']}) [{r['score']:.3f}]")
    
    # Test 2: Refinement with FAANG
    print("\n=== Test 2: at FAANG companies ===")
    history = [
        {"role": "user", "content": "machine learning jobs"},
        {"role": "assistant", "content": f"Understood intent: {json.dumps(intent)}"}
    ]
    intent2 = engine.parse_query("at FAANG companies", history)
    print("Intent:", json.dumps(intent2, indent=2))
    results2 = engine.search(intent2, top_k=5)
    for r in results2:
        print(f"  {r['title']} @ {r['company']} ({r['workplace_type']}) [{r['score']:.3f}]")
    
    print("\n" + engine.tracker.report())
