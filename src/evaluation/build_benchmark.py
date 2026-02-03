import io
import json
import sys
from datetime import datetime
from pathlib import Path


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gbk', errors='replace')

def extract_date_from_filename(filename):
    """Extract date from report filename in format: XXXXXX_YYYY-MM-DD_title.pdf"""
    try:
        parts = filename.split('_')
        if len(parts) >= 2:
            date_str = parts[1]
            return datetime.strptime(date_str, '%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing date from {filename}: {e}")
    return None


def filter_and_sort_comparison_results():
    # Read the comparison_results.json file
    data_path = Path(__file__).parent.parent.parent / 'data' / 'output' / 'comparison_results.json'

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total items in original data: {len(data)}")

    # Step 1: Filter by common_evidence_count > 50
    filtered_data = [item for item in data if item['common_evidence_count'] > 50]
    print(f"\nAfter filtering common_evidence_count > 50: {len(filtered_data)} items")

    # Step 2: Filter by date difference >= 1 month
    date_filtered_data = []
    for item in filtered_data:
        old_date = extract_date_from_filename(item['old_report'])
        new_date = extract_date_from_filename(item['new_report'])

        if old_date and new_date:
            # Calculate the difference in days
            date_diff = abs((new_date - old_date).days)
            # At least 1 month (approximately 30 days)
            if date_diff >= 31:
                date_filtered_data.append(item)

    print(f"After filtering date difference >= 1 month: {len(date_filtered_data)} items")

    # Step 2.5: Filter by new_date range (2025-09-01 to 2025-12-31)
    start_date = datetime(2025, 9, 1)
    end_date = datetime(2025, 12, 31)
    date_range_filtered_data = []
    for item in date_filtered_data:
        new_date = extract_date_from_filename(item['new_report'])
        if new_date and start_date <= new_date <= end_date:
            date_range_filtered_data.append(item)

    print(f"After filtering new_date between 2025-09-01 and 2025-12-31: {len(date_range_filtered_data)} items")

    # Step 3: Calculate ratio and sort in descending order
    results = []
    for item in date_range_filtered_data:
        if item['new_evidence_count'] > 0:
            ratio = item['common_evidence_count'] / item['new_evidence_count']
            results.append({
                'stock_code': item['stock_code'],
                'old_report': item['old_report'],
                'new_report': item['new_report'],
                'common_evidence_count': item['common_evidence_count'],
                'new_evidence_count': item['new_evidence_count'],
                'ratio': ratio
            })

    # Sort by ratio in descending order
    results.sort(key=lambda x: x['ratio'], reverse=True)

    # Print results
    print(f"\n{'='*100}")
    print(f"Final Results (sorted by ratio in descending order):")
    print(f"{'='*100}\n")

    for idx, item in enumerate(results, 1):
        print(f"{idx}. Stock Code: {item['stock_code']}")
        print(f"   Old Report: {item['old_report']}")
        print(f"   New Report: {item['new_report']}")
        print(f"   Common Evidence Count: {item['common_evidence_count']}")
        print(f"   New Evidence Count: {item['new_evidence_count']}")
        print(f"   Ratio (common/new): {item['ratio']:.4f} ({item['ratio']*100:.2f}%)")
        print()

    # Save top 50 to benchmark.json
    benchmark_data = []
    for item in results[:20]:
        new_date = extract_date_from_filename(item['new_report'])
        benchmark_item = {
            'stock_code': item['stock_code'],
            'date': new_date.strftime('%Y-%m-%d'),
            'reference': item['old_report'],
            'test': item['new_report']
        }
        benchmark_data.append(benchmark_item)

    # Save to benchmark.json
    benchmark_path = Path(__file__).parent.parent.parent / 'benchmark.json'
    with open(benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*100}")
    print(f"Saved top 50 results to {benchmark_path}")
    print(f"{'='*100}\n")

    return results


if __name__ == '__main__':
    filter_and_sort_comparison_results()
