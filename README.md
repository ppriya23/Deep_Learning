# Driving theory RAG Chatbot

Designed as an AI powered study assistant for Swedish driving theory learners, the chatbot retrieves information from the "Introduction" section of the 2025 Driving Theory Book. It delivers structured, accurate explanations of foundational driving concepts to support self-paced learning. Coverage is limited to the introduction section, learner should verify details from full book. 

- The chat bot uses an API key stored as an environment variable api_key = os.getenv("API_KEY") for secure authentication.
- Utilizes sentence-based chunking, embeddings, and semantic search for information retrieval.
- Vector storage enables efficient querying within the introduction section.
- Performance could improve by expanding coverage to the full driving theory book. 
