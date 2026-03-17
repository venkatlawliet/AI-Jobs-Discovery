import sys
import json
import argparse
from data_loader import JobsDataLoader
from search_engine import JobSearchEngine

def print_separator():
    print("\n" + "="*80 + "\n")

def format_salary(sal_min, sal_max):
    if sal_min and sal_max:
        return f"${sal_min:,.0f} - ${sal_max:,.0f}"
    elif sal_min:
        return f"${sal_min:,.0f}+"
    elif sal_max:
        return f"Up to ${sal_max:,.0f}"
    return "Not listed"

def print_results(results):
    if not results:
        print("No matching jobs found. Try adjusting your query or removing filters.")
        return
    
    print(f"\n✅ Top {len(results)} Results:\n")
    for i, res in enumerate(results, 1):
        title = res.get('title', 'Unknown Title')
        company = res.get('company', 'Unknown Company')
        location = res.get('location', 'Unknown Location')
        workplace = res.get('workplace_type', 'Unknown')
        seniority = res.get('seniority', 'Unknown')
        salary = format_salary(res.get('salary_min'), res.get('salary_max'))
        score = res.get('score', 0)
        
        print(f"{i}. {title}")
        print(f"   🏢 {company} | 📍 {location} ({workplace})")
        print(f"   📊 {seniority} | 💰 {salary}")
        print(f"   🔗 {res.get('apply_url', '')}  [Relevance: {score:.3f}]\n")

def main():
    parser = argparse.ArgumentParser(description="AI Job Search Engine Demo")
    parser.add_argument("--max_rows", type=int, default=10000, 
                        help="Maximum number of jobs to load (default: 10000)")
    parser.add_argument("--file", type=str, default="jobs.jsonl",
                        help="Path to jobs.jsonl file")
    parser.add_argument("--auto", action="store_true",
                        help="Run the automated multi-turn refinement demo")
    args = parser.parse_args()

    print_separator()
    print("Welcome to the AI Job Search Prototype!")
    print(f"Loading up to {args.max_rows} jobs from {args.file}. Please wait...")

    try:
        loader = JobsDataLoader(file_path=args.file, max_rows=args.max_rows)
        meta, me, mi, mc = loader.load()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print("\nInitializing Search Engine...")
    engine = JobSearchEngine(meta, me, mi, mc)
    print("Search Engine Ready!")
    print_separator()

    conversation_history = []
    
    if args.auto:
        # Multiple demo sessions showing search + refinement capabilities.
        # Includes the exact refinement flow from the spec (Session 1),
        # plus queries that showcase ambiguity handling and intent parsing.
        demo_sessions = [
            {
                "name": "Session 1: Data Science → Social Good → Remote (from spec)",
                "queries": [
                    "data science jobs",
                    "at companies or non-profits that care about social good",
                    "make it remote"
                ]
            },
            {
                "name": "Session 2: ML Jobs → FAANG Companies",
                "queries": [
                    "machine learning jobs",
                    "at FAANG companies"
                ]
            },
            {
                "name": "Session 3: Ambiguity — 'senior but not management'",
                "queries": [
                    "something senior but not management"
                ]
            },
            {
                "name": "Session 4: Intent — 'remote-friendly startups'",
                "queries": [
                    "remote-friendly startups in healthcare"
                ]
            },
            {
                "name": "Session 5: Senior SWE with salary",
                "queries": [
                    "senior software engineer remote"
                ]
            }
        ]
        
        for session in demo_sessions:
            print(f"\n{'='*60}")
            print(f"  📋 {session['name']}")
            print(f"{'='*60}")
            conversation_history = []  # Reset per session
            
            for user_input in session['queries']:
                print(f"\nSearch > {user_input}")
                print("\n⏳ Understanding your intent...")
                parsed_intent = engine.parse_query(user_input, conversation_history)
                print(f"🧠 Parsed Intent:")
                print(f"   Explicit: {parsed_intent.get('explicit_query', '')}")
                print(f"   Inferred: {parsed_intent.get('inferred_query', '')}")
                print(f"   Company:  {parsed_intent.get('company_query', '')}")
                print(f"   Filters:  {json.dumps(parsed_intent.get('filters', {}))}")
                
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": f"Understood intent: {json.dumps(parsed_intent)}"})
                
                print("\n🔍 Searching...")
                results = engine.search(parsed_intent, top_k=5)
                print_results(results)
            
            print_separator()
        
        print(engine.tracker.report())
        print_separator()
        return

    print("You can search for jobs like 'data science roles' and refine by saying 'make it remote'.")
    print("Type 'new' to start a fresh conversation, 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Search > ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['exit', 'quit']:
            break
        
        if user_input.lower() == 'new':
            conversation_history = []
            print("🔄 Starting a new conversation.\n")
            continue
            
        if not user_input.strip():
            continue

        print("\n⏳ Understanding your intent...")
        parsed_intent = engine.parse_query(user_input, conversation_history)
        print(f"🧠 Parsed Intent:")
        print(f"   Explicit: {parsed_intent.get('explicit_query', '')}")
        print(f"   Inferred: {parsed_intent.get('inferred_query', '')}")
        print(f"   Company:  {parsed_intent.get('company_query', '')}")
        print(f"   Filters:  {json.dumps(parsed_intent.get('filters', {}))}")
        
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": f"Understood intent: {json.dumps(parsed_intent)}"})
        
        print("\n🔍 Searching...")
        results = engine.search(parsed_intent, top_k=10)
        print_results(results)
            
        print_separator()
        print(engine.tracker.report())
        print_separator()

if __name__ == '__main__':
    main()
