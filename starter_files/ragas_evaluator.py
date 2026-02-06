from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1"
    )
    evaluator = LangchainLLMWrapper(llm)
    ellm = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1"
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(ellm)
    print("\n\n\n\n\nLOOOOOOOOOOOOOOKKKKK\n\n\n",answer, question, contexts)
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer,
        reference_contexts=contexts
    )
    ragas_dataset = EvaluationDataset(samples=[sample])
    dataset = evaluate(
        dataset=ragas_dataset,
        metrics=[
            BleuScore(), 
            NonLLMContextPrecisionWithReference(), 
            ResponseRelevancy(), 
            Faithfulness(), 
            RougeScore()
        ],
        llm=evaluator,
        embeddings=evaluator_embeddings
    )
    return dataset.to_pandas().to_dict()
