import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch


class RAGEvaluator:
    def __init__(self, model_name="mistral"):  
        self.llm_model = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"RAG using LLM: {self.llm_model}")
        print(f"Embeddings on: {self.device}")

    
        self.embedding_model = SentenceTransformer(
            "intfloat/e5-large-v2",
            device=self.device
        )


    def query_llm(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return ""

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_passages(self, passages: List[str]) -> np.ndarray:
        """Batch embed passages ONCE (GPU-optimized)"""
        return self.embedding_model.encode(
            passages,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
    
    def answer_correctness(self, question: str, ground_truth: str, answer: str) -> float:
        prompt = f"""
            Evaluate how correct the answer is compared to the ground truth.

            Question: {question}
            Ground Truth: {ground_truth}
            Answer: {answer}

            Score the answer from 0 to 1.
            Provide only the numerical score:
            """
        response = self.query_llm(prompt)
        try:
            return float(np.clip(float(response), 0, 1))
        except:
            gt_emb = self.get_embedding(ground_truth)
            ans_emb = self.get_embedding(answer)
            return self.cosine_similarity(gt_emb, ans_emb)

    def response_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""
            Evaluate how relevant the answer is to the question.

            Question: {question}
            Answer: {answer}

            Score from 0 to 1.
            Provide only the numerical score:
            """
        response = self.query_llm(prompt)
        try:
            return float(np.clip(float(response), 0, 1))
        except:
            q_emb = self.get_embedding(question)
            a_emb = self.get_embedding(answer)
            return self.cosine_similarity(q_emb, a_emb)

    def factual_correctness(self, answer: str, context: str) -> float:
        prompt = f"""
            Evaluate factual correctness using the context.

            Context: {context}
            Answer: {answer}

            Score from 0 to 1.
            Provide only the numerical score:
            """
        response = self.query_llm(prompt)
        try:
            return float(np.clip(float(response), 0, 1))
        except:
            ans_emb = self.get_embedding(answer)
            ctx_emb = self.get_embedding(context)
            return 0.8 * self.cosine_similarity(ans_emb, ctx_emb)

    def context_recall(self, ground_truth: str, retrieved_context: str) -> float:
        gt_emb = self.get_embedding(ground_truth)
        ctx_emb = self.get_embedding(retrieved_context)
        return self.cosine_similarity(gt_emb, ctx_emb)

    def evaluate_rag(
        self,
        questions: List[str],
        ground_truths: List[str],
        passages: List[str],
        documents: List[str]
    ) -> Dict[str, float]:

        results = {
            "answer_correctness": [],
            "response_relevancy": [],
            "factual_correctness": [],
            "context_recall": [],
        }


        passage_embeddings = self.embed_passages(passages)

        for question, ground_truth in zip(questions, ground_truths):
            q_emb = self.get_embedding(question)

            sim_scores = passage_embeddings @ q_emb

            top_idx = np.argsort(sim_scores)[-3:][::-1]
            context = " ".join([passages[i] for i in top_idx])

            prompt = f"""
                Answer the question using ONLY the context below.

                Context:
                {context}

                Question: {question}
                Answer:
                """
            answer = self.query_llm(prompt)

            results["answer_correctness"].append(
                self.answer_correctness(question, ground_truth, answer)
            )
            results["response_relevancy"].append(
                self.response_relevancy(question, answer)
            )
            results["factual_correctness"].append(
                self.factual_correctness(answer, context)
            )
            results["context_recall"].append(
                self.context_recall(ground_truth, context)
            )

        return {k: float(np.mean(v)) for k, v in results.items()}