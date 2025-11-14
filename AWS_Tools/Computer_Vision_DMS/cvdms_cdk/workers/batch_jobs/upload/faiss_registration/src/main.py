import os
import json
import boto3
import faiss
import numpy as np

s3 = boto3.client("s3")

BUCKET = os.environ["FILE_BUCKET_NAME"]
INDEX_PREFIX = "canonical/faiss_indices"

def load_index_from_s3(key: str):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = obj["Body"].read()
        return faiss.deserialize_index(data)
    except s3.exceptions.NoSuchKey:
        return None

def save_index_to_s3(index, key: str):
    data = faiss.serialize_index(index)
    s3.put_object(Bucket=BUCKET, Key=key, Body=data)

def handler():
    # Step Functions Batch integration passes the state input as env var JSON
    manifest_str = os.environ.get("MANIFESTS")
    if not manifest_str:
        raise ValueError("No MANIFESTS provided")

    manifests = json.loads(manifest_str)

    # Extract phashes (assume hex strings) and convert to vectors
    new_phashes = [m["phash"] for m in manifests]
    vectors = np.array([int(p, 16) for p in new_phashes], dtype="int64")
    vectors = vectors.reshape(-1, 1).astype("float32")  # FAISS expects float32

    # Load existing index if present
    index_key = f"{INDEX_PREFIX}/main.index"
    index = load_index_from_s3(index_key)

    if index is None:
        # First time: build new index
        index = faiss.IndexFlatL2(1)  # 1D vectors (phash as scalar)
        index.add(vectors)
    else:
        index.add(vectors)

    # Save updated index
    save_index_to_s3(index, index_key)

    # Optional: shard if too large
    if index.ntotal > 1000000:  # example threshold
        # Split by phash prefix (0/1)
        vecs = vectors.flatten().astype("int64")
        shard0 = [v for v in vecs if (v & 1) == 0]
        shard1 = [v for v in vecs if (v & 1) == 1]

        for shard_name, shard_vecs in [("0", shard0), ("1", shard1)]:
            if shard_vecs:
                arr = np.array(shard_vecs, dtype="int64").reshape(-1, 1).astype("float32")
                shard_index = faiss.IndexFlatL2(1)
                shard_index.add(arr)
                save_index_to_s3(shard_index, f"{INDEX_PREFIX}/shard_{shard_name}.index")

    print(f"Processed {len(new_phashes)} new images, index size now {index.ntotal}")

if __name__ == "__main__":
    handler()
