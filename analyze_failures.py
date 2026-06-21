import pandas as pd

results = pd.read_csv('data/track6_evaluation_results_local_llm.csv')
failing = results[results['critical_entity_preservation_ok'] == False]

print(f"\n=== Remaining Failing Summaries ({len(failing)} out of 15) ===\n")
for idx, row in failing.iterrows():
    print(f"Report: {row['report_id']}")
    print(f"  Word count: {row['word_count']} (pass: {row['length_ok']})")
    print(f"  Customer: {row['has_customer']}, Counterparty: {row['has_counterparty']}")
    print(f"  Banks: {row['has_banks']}, Amount: {row['has_amount']}")
    print(f"  Expected Banks: {row['entities_bank_names']}")
    print()

passing = results[results['critical_entity_preservation_ok'] == True]
print(f"\n=== Passing Summaries ({len(passing)} out of 15) ✓ ===\n")
for idx, row in passing.iterrows():
    print(f"  {row['report_id']}")

# Show bank name issue detail
print("\n=== Example: RPT-2026-000002 (Banks: False) ===\n")
row = results[results['report_id'] == 'RPT-2026-000002'].iloc[0]
print(f"Expected Banks: {row['entities_bank_names']}")
print(f"\nSummary snippet: ...{row['summary'][80:280]}...")
