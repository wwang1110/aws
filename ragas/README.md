# ragas-lab
RAG (Retrieval-Augmented Generation) pipelines consist of two key components:
1. Retriever: Responsible for extracting the most pertinent information to address the query.
2. Generator: Tasked with formulating a response using the retrieved information.
To effectively evaluate a RAG pipeline, it's crucial to assess these components both individually and collectively. This approach yields an overall performance score while also providing specific metrics for each component, allowing for targeted improvements. For instance:
- Enhancing the Retriever: This can be achieved through improved chunking strategies or by employing more advanced embedding models.
- Optimizing the Generator: Experimenting with different language models or refining prompts can lead to better generation outcomes.
However, this raises several important questions: What metrics should be used to measure and benchmark these components? Which datasets are most suitable for evaluation? How can Amazon Bedrock be integrated with RAGAS for this purpose?
In the following labs, we'll delve into these critical aspects and show you how to use a framework called RAGAS to create RAG pipeline evaluation and optimization.
