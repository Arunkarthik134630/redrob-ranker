# run.py - ranks candidates and fixes tie-break in one shot
import subprocess
import csv
import sys

# Step 1: Run the ranker
print("=== Running ranker ===")
subprocess.run([sys.executable, "rank.py", 
                "--candidates", "candidates.jsonl", 
                "--out", "outputs\\submission.csv"], check=True)

# Step 2: Fix tie-break
print("\n=== Fixing tie-break ===")
rows = []
with open(r'outputs\submission.csv', encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: (-float(r['score']), r['candidate_id']))

with open(r'outputs\submission.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    prev = None
    for i, r in enumerate(rows):
        s = float(r['score'])
        if prev is not None and s > prev:
            s = prev
        prev = s
        w.writerow([r['candidate_id'], i+1, f'{s:.4f}', r['reasoning']])

print("Tie-break fixed!")

# Step 3: Validate
print("\n=== Validating ===")
subprocess.run([sys.executable, "validate_submission.py", 
                "outputs\\submission.csv"], check=True)