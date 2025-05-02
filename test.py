import subprocess
import pandas as pd

def check_hit(predicted_ids, ground_truth_ids, top_k=10):
    predicted_topk = predicted_ids[:top_k]
    ground_truth_set = set(ground_truth_ids)
    predicted_set = set(predicted_topk)
    return 1 if predicted_set & ground_truth_set else 0

# Load test.csv
df = pd.read_csv('test.csv')

hits = 0
total_queries = len(df)
total_relevant = total_queries  # assume 1 ground truth per query

for idx, row in df.iterrows():
    query = row['query']
    ground_truth_ids = [str(row['related_doc_ids'])] if isinstance(row['related_doc_ids'], int) else str(row['related_doc_ids']).split(',')

    # Call user_new.py with subprocess
    process = subprocess.Popen(
        ['python', 'user_new.py', query],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    output = stdout.decode('utf-8')

    # Extract predicted doc_ids from output lines like “[180#1] similarity=0.532”
    predicted_ids = []
    for line in output.splitlines():
        if line.startswith('['):
            doc_id = line.split(']')[0][1:].split('#')[0]
            predicted_ids.append(doc_id)

    # Evaluate hit (1) or miss (0)
    hit = check_hit(predicted_ids, ground_truth_ids)
    hits += hit
    print(f"Query ID {row['qid']} → Hit: {hit}")

# Compute overall precision and recall
# In this setup:
# precision = hits / total predictions (top_k * total_queries)
# recall = hits / total relevant (same as hits / total_queries if one ground truth per query)

precision = hits / (total_queries)  # treating one prediction per query (the hit decision)
recall = hits / total_relevant

print(f"\nOverall Precision: {precision:.3f}")
print(f"Overall Recall: {recall:.3f}")
