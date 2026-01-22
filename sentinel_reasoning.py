import time
import torch
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from transformers import CLIPProcessor, CLIPModel
import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
COLLECTION_NAME = "sentinel_vision"
client = QdrantClient(path="./qdrant_data")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def get_text_embedding(text):
    """Converts a text description into a vector."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize for Cosine Similarity
    return (text_features / text_features.norm(p=2, dim=-1, keepdim=True))[0].tolist()
def calculate_decay(base_risk, elapsed_seconds):
    """
    Decays risk over time. 
    """
    decay_factor = 0.5 ** (elapsed_seconds / 60) 
    return base_risk * decay_factor
def assess_sector(sector_name, description):
    print(f"Context: '{description}'")

    # SEARCH MEMORY 
    query_vector = get_text_embedding(description)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1
    )
    # Access the points inside the result
    hits = results.points
    if not hits:
        print("No memory correlation found.")
        return
    top_hit = hits[0]
    payload = top_hit.payload
    # APPLY REASONING
    base_risk = payload['risk_score']
    stored_time = payload.get('timestamp', time.time())
    elapsed_seconds = time.time() - stored_time
    final_risk = calculate_decay(base_risk, elapsed_seconds)
    print(f"EVIDENCE: Matched '{payload['hazard_type']}' image")
    print(f"(Source: {payload['source']})")
    print(f"(Time Decay: {(elapsed_seconds)}s elapsed)")
    print(f"(Similarity: {top_hit.score})")
    # DECISION MATRIX
    if final_risk > 0.5:
        print(f"DECISION: DANGER DETECTED (Risk: {final_risk})")
        print(f"ACTION: Reroute Autonomous Agent immediately.")
    else:
        print(f"DECISION: AREA SECURE (Risk: {final_risk})")
        print(f"ACTION: Continue Search & Rescue path.")
def main():
    # Check if collection exists first
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Error: Collection {COLLECTION_NAME} not found.") 
        return
    # Scenario 1: Drone sees a nice park
    assess_sector("Sector Alpha", "a peaceful grassy path with trees")
    # Scenario 2: Drone reports massive heat
    assess_sector("Sector Beta", "huge raging fire and smoke")
    # Scenario 3: Drone reports water levels rising
    assess_sector("Sector Gamma", "houses submerged in deep water")
if __name__ == "__main__":
    main()