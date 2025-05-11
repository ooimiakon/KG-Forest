import pandas as pd
from simple_embedding import retrieve_top_k  # Make sure this exists and is importable

# Load test set
df = pd.read_csv('test.csv')

top_k = 5
hits = 0

for idx, row in df.iterrows():
    query = row['query']
    qid = str(row['qid'])
    ground_truth_ids = [id_.strip() for id_ in str(row['related_doc_ids']).split(',')]

    # Directly retrieve top-k doc IDs
    results = retrieve_top_k(query, top_k=top_k)
    predicted_ids = [str(doc_id) for doc_id, _ in results]

    # Check for a hit
    hit = int(any(pid in ground_truth_ids for pid in predicted_ids))
    hits += hit
    predicted_ids_str = ', '.join(predicted_ids) if predicted_ids else 'N/A'

    # Print details
    print("=" * 60)
    print(f"Query ID        : {qid}")
    print(f"Query           : {query}")
    print(f"Expected Doc ID : {', '.join(ground_truth_ids)}")
    print(f"Predicted Doc IDs (top-{top_k}): {predicted_ids_str}")
    print(f"Hit             : {hit}")

# Final evaluation
total = len(df)
precision_at_k = hits / total if total > 0 else 0
print("\n" + "=" * 60)
print(f"Total Queries    : {total}")
print(f"Top-{top_k} Hits      : {hits}")
print(f"Precision@{top_k}     : {precision_at_k:.3f}")
