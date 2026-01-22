# Sentinel: AI Disaster Response Agent
**Submitted for:** Qdrant Convolve 4.0 Hackathon

## 1. Project Overview
**Sentinel** is an autonomous agent architecture designed to assist first responders in disaster scenarios. It utilizes **Qdrant** as a probabilistic semantic memory, allowing rescue drones to reason over multimodal data and make path-planning decisions based on evolving hazard levels.

This system addresses the **Disaster Response & Public Safety** societal challenge by reducing information overload and enabling autonomous reasoning over fragmented data streams.

## 2. System Architecture
The system is composed of three core modules:

* **The Senses (Ingestion):** Uses `OpenAI/CLIP-ViT-B32` to project disparate data types into a shared 512-dimensional vector space.
* **The Hippocampus (Qdrant):** Qdrant is used for **Associative Retrieval**. Unlike traditional databases, it allows the agent to query visual memories using text prompts.
* **The Reasoning Engine (Logic):** We implement a **Temporal Decay Function** on top of Qdrant's similarity scores.
    * **Formula:** Risk(t) = Similarity * e^(-lambda * t)
    * **Effect:** This ensures the agent prioritizes recent hazards over outdated information, mimicking human short-term memory.

## 3. Setup Instructions

### Prerequisites
* Python 3.8+
* 3 Test Images placed in the root directory:
    * `fire.jpg`
    * `flood.jpg`
    * `safe.jpg`

### Installation
Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```
*(Dependencies: `qdrant-client`, `torch`, `transformers`, `pillow`, `requests`)*

## 4. How to Run

### Step 1: Initialize Memory
Run the ingestion script to process the visual terrain data and store them in the local Qdrant vector database. This simulates a drone mapping an area.

```bash
python sentinel_ingest_local.py
```
* **Action:** Downloads CLIP model, embeds images, and stores vectors with metadata in `./qdrant_data`.
* **Expected Output:** `INGESTION COMPLETE.`

### Step 2: Run Agent Logic
Run the reasoning engine. The agent will query its memory using text descriptions to find visual matches, calculate risk based on real-time decay and make autonomous decisions.

```bash
python sentinel_reasoning.py
```
* **Action:** Queries Qdrant using text vectors, retrieves image evidence, applies decay logic and outputs a decision
* **Expected Output:**
    > DECISION: DANGER DETECTED
    > EVIDENCE: Matched 'fire' image

## 5. File Structure
* `sentinel_ingest_local.py`: Ingestion pipeline.
* `sentinel_reasoning.py`: Decision engine.
* `requirements.txt`: Python dependencies.
* `qdrant_data/`: Local storage for the vector database.