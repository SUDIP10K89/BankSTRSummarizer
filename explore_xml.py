"""
Explore STR XML structure and analyze narrative characteristics
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path
from collections import defaultdict
import statistics

# Configuration
REPORTS_DIR = Path(__file__).parent / "reports"
OUTPUT_FILE = Path(__file__).parent / "xml_exploration_report.txt"

def parse_report(xml_file):
    """Parse a single STR XML report and extract key fields."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        data = {
            'report_id': root.findtext('report_id', 'N/A'),
            'entity_reference': root.findtext('entity_reference', 'N/A'),
            'submission_date': root.findtext('submission_date', 'N/A'),
            'reason': root.findtext('reason', ''),
            'currency_code': root.findtext('currency_code_local', 'N/A'),
        }
        
        # Extract transaction details
        transaction = root.find('transaction')
        if transaction is not None:
            data['transaction_number'] = transaction.findtext('transactionnumber', 'N/A')
            data['amount_local'] = transaction.findtext('amount_local', '0')
            data['date_transaction'] = transaction.findtext('date_transaction', 'N/A')
            data['transmode_comment'] = transaction.findtext('transmode_comment', 'N/A')
            
            # From account details
            from_account = transaction.find('.//t_from_my_client/from_account')
            if from_account is not None:
                data['from_account_name'] = from_account.findtext('account_name', 'N/A')
                data['from_account_number'] = from_account.findtext('account', 'N/A')
            
            # To account details
            to_account = transaction.find('.//t_to/to_account')
            if to_account is not None:
                data['to_account_name'] = to_account.findtext('account_name', 'N/A')
                data['to_account_number'] = to_account.findtext('account', 'N/A')
        
        return data
    except Exception as e:
        print(f"Error parsing {xml_file.name}: {e}")
        return None

def explore_xml_structure():
    """Main exploration function."""
    
    print("🔍 Exploring STR XML Structure")
    print("=" * 80)
    
    # Collect all reports
    xml_files = sorted(REPORTS_DIR.glob("report_*.xml"))
    print(f"\n📊 Found {len(xml_files)} report files")
    
    reports_data = []
    narrative_lengths = []
    
    # Parse all reports
    print(f"\n⏳ Parsing {len(xml_files)} reports...")
    for i, xml_file in enumerate(xml_files):
        data = parse_report(xml_file)
        if data:
            reports_data.append(data)
            narrative_lengths.append(len(data['reason']))
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ Processed {i + 1} reports")
    
    print(f"\n✅ Successfully parsed {len(reports_data)} reports")
    
    # Analyze narrative lengths
    print("\n" + "=" * 80)
    print("📈 NARRATIVE LENGTH ANALYSIS")
    print("=" * 80)
    
    if narrative_lengths:
        min_len = min(narrative_lengths)
        max_len = max(narrative_lengths)
        avg_len = statistics.mean(narrative_lengths)
        median_len = statistics.median(narrative_lengths)
        stdev = statistics.stdev(narrative_lengths) if len(narrative_lengths) > 1 else 0
        
        print(f"Total narratives: {len(narrative_lengths)}")
        print(f"Min length: {min_len} characters")
        print(f"Max length: {max_len} characters")
        print(f"Mean length: {avg_len:.1f} characters")
        print(f"Median length: {median_len:.1f} characters")
        print(f"Std Dev: {stdev:.1f} characters")
        
        # Distribution
        buckets = {
            '0-50': 0,
            '51-100': 0,
            '101-200': 0,
            '201-500': 0,
            '501-1000': 0,
            '1001-2000': 0,
            '2000+': 0
        }
        
        for length in narrative_lengths:
            if length <= 50:
                buckets['0-50'] += 1
            elif length <= 100:
                buckets['51-100'] += 1
            elif length <= 200:
                buckets['101-200'] += 1
            elif length <= 500:
                buckets['201-500'] += 1
            elif length <= 1000:
                buckets['501-1000'] += 1
            elif length <= 2000:
                buckets['1001-2000'] += 1
            else:
                buckets['2000+'] += 1
        
        print("\nNarrative length distribution:")
        for bucket, count in buckets.items():
            pct = 100 * count / len(narrative_lengths)
            bar = "█" * int(pct / 2)
            print(f"  {bucket:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Sample reports of different types
    print("\n" + "=" * 80)
    print("📋 SAMPLE REPORTS")
    print("=" * 80)
    
    # Find shortest, longest, and median length reports
    if reports_data:
        sorted_by_length = sorted(reports_data, key=lambda x: len(x['reason']))
        
        print("\n1️⃣  SHORTEST NARRATIVE:")
        print("-" * 80)
        shortest = sorted_by_length[0]
        print(f"Report ID: {shortest['report_id']}")
        print(f"Length: {len(shortest['reason'])} chars")
        print(f"Reason:\n{shortest['reason'][:500]}")
        
        print("\n\n2️⃣  MEDIAN NARRATIVE:")
        print("-" * 80)
        median_report = sorted_by_length[len(sorted_by_length) // 2]
        print(f"Report ID: {median_report['report_id']}")
        print(f"Length: {len(median_report['reason'])} chars")
        print(f"Reason:\n{median_report['reason'][:800]}")
        
        print("\n\n3️⃣  LONGEST NARRATIVE:")
        print("-" * 80)
        longest = sorted_by_length[-1]
        print(f"Report ID: {longest['report_id']}")
        print(f"Length: {len(longest['reason'])} chars")
        print(f"Reason:\n{longest['reason'][:1000]}")
    
    # Analyze structured fields
    print("\n" + "=" * 80)
    print("📊 STRUCTURED FIELDS ANALYSIS")
    print("=" * 80)
    
    currency_counts = defaultdict(int)
    transmode_counts = defaultdict(int)
    
    for report in reports_data:
        currency_counts[report['currency_code']] += 1
        transmode_counts[report.get('transmode_comment', 'Unknown')] += 1
    
    print("\nTop 10 currencies:")
    for curr, count in sorted(currency_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {curr}: {count}")
    
    print("\nTop 10 transaction modes:")
    for mode, count in sorted(transmode_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {mode}: {count}")
    
    # Key facts extraction - sample
    print("\n" + "=" * 80)
    print("🔑 KEY FACTS IN NARRATIVES")
    print("=" * 80)
    
    # Look for patterns in one longer narrative
    sample_long = sorted_by_length[-1]
    print(f"\nAnalyzing: {sample_long['report_id']}")
    print("\nKey facts to preserve in summary:")
    print(f"  - Amount: {sample_long.get('amount_local', 'N/A')}")
    print(f"  - Currency: {sample_long['currency_code']}")
    print(f"  - Date: {sample_long.get('date_transaction', 'N/A')}")
    print(f"  - From: {sample_long.get('from_account_name', 'N/A')}")
    print(f"  - To: {sample_long.get('to_account_name', 'N/A')}")
    print(f"  - Transaction Mode: {sample_long.get('transmode_comment', 'N/A')}")
    
    return reports_data

if __name__ == "__main__":
    explore_xml_structure()
    print("\n" + "=" * 80)
    print("✅ Exploration complete!")
    print("=" * 80)
