import subprocess
import sys

print("Running BankSTRSummarizer pipeline\n")

scripts = [
    "data_pipeline.py",
    "entity_extraction.py",

    "llm_summarizer.py --resume --limit 10",
    "evaluation.py",

    "llm_summarizer_local.py --resume --limit 10",
    "evaluation.py --summaries data/track6_llm_summaries_local.csv --output data/track6_evaluation_results_local_llm.csv --summary-output data/track6_evaluation_summary_local_llm.json",

    "extractive_summary.py",
    "evaluation.py --summaries data/summary_extractive.csv --output data/track6_evaluation_results_extractive.csv --summary-output data/track6_evaluation_summary_extractive.json",

    "hybrid_summary.py",
    "evaluation.py --summaries data/summary_hybrid.csv --output data/track6_evaluation_results_hybrid.csv --summary-output data/track6_evaluation_summary_hybrid.json"
]

for script in scripts:
    print(f"Running {script}")
    result = subprocess.run([sys.executable] + script.split())
    
    if result.returncode != 0:
        print(f"Error running {script}")
        break

print("\nPipeline completed.")
