import csv
import os
import argparse
import sys

"""
Find segments present in the original CSV but missing from batch CSVs.

Sample command:

python3 scripts/find_missing_segments.py \
  --original "data/audioset/unbalanced_train_segments.csv" \
  --batches \
    "data/audioset/unbalanced_train_segments_batch_0.csv" \
    "data/audioset/unbalanced_train_segments_batch_1.csv" \
    "data/audioset/unbalanced_train_segments_batch_2.csv" \
    "data/audioset/unbalanced_train_segments_batch_3.csv" \
  --output "data/audioset/unbalanced_train_segments_batch_4.csv"

python3 scripts/find_missing_segments.py \
  --validate \
  --original "data/audioset/unbalanced_train_segments.csv" \
  --batches \
    "data/audioset/unbalanced_train_segments_batch_0.csv" \
    "data/audioset/unbalanced_train_segments_batch_1.csv" \
    "data/audioset/unbalanced_train_segments_batch_2.csv" \
    "data/audioset/unbalanced_train_segments_batch_3.csv" \
    "data/audioset/unbalanced_train_segments_batch_4.csv"
"""




def normalize_row(row):
    """
    Returns a tuple (video_id, start_time, end_time) for comparison.
    Strips whitespace.
    """
    if len(row) < 3:
        return None
    # row[0] is ID, row[1] is start, row[2] is end
    return (row[0].strip(), row[1].strip(), row[2].strip())

def count_rows(filepath):
    """Counts data rows in a CSV, skipping headers and comments."""
    count = 0
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return 0
        
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            # Skip comments
            if row[0].strip().startswith("#"):
                continue
            # Skip header
            if row[0].strip().lower() == "video_id":
                continue
            count += 1
    return count

def validate_counts(original_path, batch_paths):
    print(f"Validating counts...")
    original_count = count_rows(original_path)
    print(f"Original file rows: {original_count}")
    
    total_batch_count = 0
    for batch_path in batch_paths:
        c = count_rows(batch_path)
        print(f"  {batch_path}: {c}")
        total_batch_count += c
        
    print(f"Total rows in batches: {total_batch_count}")
    
    if original_count == total_batch_count:
        print("SUCCESS: Counts match!")
        return True
    else:
        diff = original_count - total_batch_count
        print(f"FAILURE: Counts do not match. Difference: {diff}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Find segments in original CSV that are missing from batch CSVs.")
    parser.add_argument("--original", required=True, help="Path to the original large CSV file")
    parser.add_argument("--batches", nargs='+', required=True, help="Paths to the batch CSV files (e.g. batch_*.csv)")
    parser.add_argument("--output", help="Path to the output CSV file (the 5th batch)")
    parser.add_argument("--validate", action="store_true", help="Validate row counts instead of generating file")
    
    args = parser.parse_args()

    if args.validate:
        validate_counts(args.original, args.batches)
        return

    if not args.output:
        parser.error("the following arguments are required: --output")
    
    # 1. Load IDs from batches
    seen_ids = set()
    print(f"Reading {len(args.batches)} batch files...")
    
    for batch_path in args.batches:
        if not os.path.exists(batch_path):
            print(f"Warning: Batch file {batch_path} not found. Skipping.")
            continue
            
        with open(batch_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # Skip headers if present (heuristic: "video_id" or starts with "#")
                if row[0].strip().lower() == "video_id" or row[0].strip().startswith("#"):
                    continue
                
                key = normalize_row(row)
                if key:
                    seen_ids.add(key)
    
    print(f"Found {len(seen_ids)} unique segments in batches.")
    
    # 2. Scan original file
    missing_rows = []
    print(f"Scanning original file: {args.original}")
    
    if not os.path.exists(args.original):
        print(f"Error: Original file {args.original} not found.")
        sys.exit(1)

    total_rows = 0
    with open(args.original, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            
            # Skip comments
            if row[0].strip().startswith("#"):
                continue
            
            # Skip header if it looks like "video_id" (though raw AudioSet usually uses # comments for header)
            if row[0].strip().lower() == "video_id":
                continue

            total_rows += 1
            key = normalize_row(row)
            
            if key:
                if key not in seen_ids:
                    # Clean the row data similar to download_audio_segments.py
                    video_id = row[0].strip()
                    start_time = row[1].strip()
                    end_time = row[2].strip()
                    
                    # Reconstruct label
                    label = ",".join(row[3:]) if len(row) > 3 else ""
                    label = label.strip()
                    
                    # Remove surrounding quotes if present
                    if label.startswith('"') and label.endswith('"'):
                        label = label[1:-1]
                        
                    missing_rows.append([video_id, start_time, end_time, label])
    
    print(f"Scanned {total_rows} data rows in original file.")
    print(f"Found {len(missing_rows)} missing rows.")
    
    # 3. Write output
    print(f"Writing missing rows to {args.output}...")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write a standard header compatible with the downloader script
        writer.writerow(["video_id", "start_time", "end_time", "label"])
        writer.writerows(missing_rows)
        
    print("Done.")

if __name__ == "__main__":
    main()
