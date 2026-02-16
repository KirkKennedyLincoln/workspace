from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os
import json
import rag_client

    

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def run_test_questions(collection, openai_key: str, n_docs: int = 3, 
                       model: str = "gpt-3.5-turbo") -> List[Dict[str, float]]:
    from rag_client import retrieve_documents, format_context
    from llm_client import generate_response
    my_tests = []
    with open('questions.json', 'r') as f:
        data = json.load(f)
        context = ""
        contexts_list = []
        for topic in data.values():
            try:
                docs_result = retrieve_documents(
                    collection, 
                    topic["question"], 
                    n_docs
                )

                if docs_result and docs_result.get("documents"):
                    context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                    contexts_list = docs_result["documents"][0]

            except Exception as e:
                print(e)
            my_message = generate_response(openai_key=openai_key, user_message=topic["question"], context=context, 
                    conversation_history=[], model="gpt-3.5-turbo")
            my_tests.append({
                "metrics": response_quality(topic["question"], my_message, contexts_list),
                "QA": {
                    "question": topic["question"],
                    "answer": topic["answer"],
                    "generated_response": my_message
                }
            })
    return my_tests

def response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
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
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        # reference=answer,
        # reference_contexts=contexts
    )
    ragas_dataset = EvaluationDataset(samples=[sample])
    dataset = evaluate(
        dataset=ragas_dataset,
        metrics=[
            # BleuScore(), 
            LLMContextPrecisionWithoutReference(), 
            ResponseRelevancy(), 
            Faithfulness(), 
            # RougeScore()
        ],
        llm=evaluator,
        embeddings=evaluator_embeddings
    )

    return dataset.to_pandas().to_dict()

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:

    return response_quality(question, answer, contexts)

if __name__ == "__main__":
    collection, success, error = rag_client.initialize_rag_system(
        "./chroma_db_openai", "nasa_space_missions_text"
    )
    key = os.environ.get("OPENAI_API_KEY")
    results = run_test_questions(collection, key)

    precision_scores = []
    relevancy_scores = []
    faithfulness_scores = []
    finish_dict = {}
    for i, r in enumerate(results):
        finish_dict[r["QA"]['question']] = r["QA"]['answer']
        print(f"\n[{r["QA"]['question']}] {r["QA"]['answer']}\n")
        print(f"""
            LLM Context Precision Without Reference: {r["metrics"]['llm_context_precision_without_reference'][0]}\n
            Answer Relevancy: {r["metrics"]['answer_relevancy'][0]}\n
            Faithfulness: {r["metrics"]['faithfulness'][0]}\n
        """)

        precision_scores.append(r["metrics"]['llm_context_precision_without_reference'][0])
        relevancy_scores.append(r["metrics"]['answer_relevancy'][0])
        faithfulness_scores.append(r["metrics"]['faithfulness'][0])


    print(f"""
            ======OVERALL STATISTICS======
            LLM Context Precision Without Reference: {sum(precision_scores)/len(precision_scores)}\n
            Answer Relevancy: {sum(relevancy_scores)/len(relevancy_scores)}\n
            Faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores)}\n
        """)
    with open('./answers.json', 'w') as f:
        json.dump(finish_dict, f)

    
