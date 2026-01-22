import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
import time
# Configuration
COLLECTION_NAME = "sentinel_vision"
MODEL_ID = "openai/clip-vit-base-patch32"
DATA_FOLDER = "."  # Current folder
def main():
    # Initialize Models
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("Vision Model Loaded.")
    # Initialize Qdrant
    client = QdrantClient(path="./qdrant_data") 
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    print("Qdrant Memory Ready.")
    test_data = [
        {"file": "fire.jpg", "type": "fire", "risk": 0.9},
        {"file": "flood.jpg", "type": "flood", "risk": 0.8},
        {"file": "safe.jpg", "type": "safe", "risk": 0.1}
    ]
    for idx, item in enumerate(test_data):
        file_path = os.path.join(DATA_FOLDER, item["file"])
        print(f"Processing: {item['type']} (looking for {item['file']})...")
        try:
            # Open Local Image
            if not os.path.exists(file_path):
                print(f"MISSING FILE: {item['file']}'")
                continue
            image = Image.open(file_path)
            # Generate Embedding
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()
            # 3. Store in Qdrant
            simulated_age = idx * 600
            fake_timestamp = time.time() - simulated_age
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=idx, 
                        vector=vector,
                        payload={
                            "hazard_type": item["type"],
                            "risk_score": item["risk"],
                            "source": item["file"], 
                            "timestamp": fake_timestamp
                        }
                    )
                ]
            )
            print(f"Stored {item['type']} (Simulated Age: {simulated_age}s ago)")
        except Exception as e:
            print(f"Error processing {item['type']}: {e}")
    print("INGESTION COMPLETE.")
if __name__ == "__main__":
    main()