import json
import os
import numpy as np

class JobsDataLoader:
    """
    Loads the jobs.jsonl dataset, extracting:
    - Rich structured metadata for filtering (company name, seniority, sector, salary, etc.)
    - Three 1536-dim embedding vectors per job for semantic search
    """
    def __init__(self, file_path='jobs.jsonl', max_rows=None):
        self.file_path = file_path
        self.max_rows = max_rows

    def _resolve_file_path(self):
        if os.path.exists(self.file_path):
            return self.file_path
        dir_name = os.path.dirname(os.path.abspath(self.file_path)) or '.'
        for f in os.listdir(dir_name):
            if f.endswith('.crdownload'):
                return os.path.join(dir_name, f)
        raise FileNotFoundError(f"Could not find {self.file_path}")

    def load(self):
        resolved_path = self._resolve_file_path()
        print(f"Loading data from: {resolved_path}")
        
        metadata = []
        explicit_vectors = []
        inferred_vectors = []
        company_vectors = []

        count = 0
        with open(resolved_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_rows and count >= self.max_rows:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    job = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # ---- Extract Vectors ----
                v7 = job.get('v7_processed_job_data', {})
                emb_explicit = v7.get('embedding_explicit_vector')
                emb_inferred = v7.get('embedding_inferred_vector')
                emb_company = v7.get('embedding_company_vector')
                
                if not emb_explicit or not emb_inferred or not emb_company:
                    continue
                
                # ---- Extract RICH Metadata ----
                info = job.get('job_information', {})
                v5_job = job.get('v5_processed_job_data', {})
                v5_company = job.get('v5_processed_company_data', {})
                work_arrangement = v7.get('work_arrangement', {})
                
                # Core fields
                job_id = job.get('id')
                title = v5_job.get('core_job_title') or info.get('title') or "Unknown Title"
                company = v5_job.get('company_name') or v5_company.get('name') or info.get('company_info', {}).get('name') or "Unknown Company"
                apply_url = job.get('apply_url')
                
                # Workplace
                workplace_type = work_arrangement.get('workplace_type') or v5_job.get('workplace_type') or "Unknown"
                locations = work_arrangement.get('workplace_locations', [])
                location_str = "Unknown"
                if locations:
                    loc = locations[0]
                    city = loc.get('city', '')
                    state = loc.get('state', '')
                    location_str = f"{city}, {state}".strip(", ") if city or state else "Unknown"
                elif v5_job.get('formatted_workplace_location'):
                    location_str = v5_job.get('formatted_workplace_location')
                
                # Seniority & Role
                seniority = v5_job.get('seniority_level', 'Unknown')
                job_category = v5_job.get('job_category', 'Unknown')
                role_type = v5_job.get('role_type', 'Unknown')  # People Manager / Individual Contributor
                commitment = v5_job.get('commitment', [])
                commitment_str = ', '.join(commitment) if isinstance(commitment, list) else str(commitment)
                
                # Company details
                company_sector = v5_job.get('company_sector_and_industry', 'Unknown')
                is_non_profit = v5_company.get('is_non_profit', False)
                is_public = v5_company.get('is_public_company', False)
                num_employees = v5_company.get('num_employees')
                company_industries = v5_company.get('industries', [])
                
                # Salary
                yearly_min = v5_job.get('yearly_min_compensation')
                yearly_max = v5_job.get('yearly_max_compensation')
                
                # Search-optimized: lowercase fields for fast matching
                company_lower = company.lower().strip()
                location_lower = location_str.lower().strip()
                
                # Extract structured state list for state-level filtering
                workplace_states = v5_job.get('workplace_states', [])
                # Normalize: "Illinois, US" -> "illinois"
                states_lower = [s.split(',')[0].strip().lower() for s in workplace_states if s]

                meta = {
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "company_lower": company_lower,
                    "apply_url": apply_url,
                    "workplace_type": workplace_type,
                    "location": location_str,
                    "seniority": seniority,
                    "job_category": job_category,
                    "role_type": role_type,
                    "commitment": commitment_str,
                    "company_sector": company_sector,
                    "is_non_profit": is_non_profit,
                    "is_public": is_public,
                    "num_employees": num_employees,
                    "company_industries": company_industries,
                    "salary_min": yearly_min,
                    "salary_max": yearly_max,
                    "location_lower": location_lower,
                    "states_lower": states_lower,
                }
                
                metadata.append(meta)
                explicit_vectors.append(emb_explicit)
                inferred_vectors.append(emb_inferred)
                company_vectors.append(emb_company)
                
                count += 1
                if count % 10000 == 0:
                    print(f"  Loaded {count} jobs...")

        print(f"Successfully loaded {len(metadata)} jobs.")
        
        mat_explicit = np.array(explicit_vectors, dtype=np.float32)
        mat_inferred = np.array(inferred_vectors, dtype=np.float32)
        mat_company = np.array(company_vectors, dtype=np.float32)
        
        return metadata, mat_explicit, mat_inferred, mat_company

if __name__ == '__main__':
    loader = JobsDataLoader(max_rows=100)
    meta, m_exp, m_inf, m_comp = loader.load()
    print(f"Shape: E={m_exp.shape}, I={m_inf.shape}, C={m_comp.shape}")
    print("Sample:", json.dumps(meta[0], indent=2, default=str))
