import ollama
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any

class RAGEvaluator:
    def __init__(self, model_name="llama3"):
        self.llm_model = model_name
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def query_llm(self, prompt: str, temperature: float = 0.1) -> str:
      
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': temperature}
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return ""
    
    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, normalize_embeddings=True)
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))
    
    def answer_correctness(self, question: str, ground_truth: str, answer: str) -> float:
        prompt = f"""Evaluate how correct the answer is compared to the ground truth.
        
        Question: {question}
        Ground Truth: {ground_truth}
        Answer: {answer}
        
        Score the answer on a scale from 0 to 1, where 1 is perfectly correct and 0 is completely wrong.
        Consider both factual accuracy and semantic similarity.
        
        Provide only the numerical score:"""
        
        response = self.query_llm(prompt)
        try:
            score = float(response.strip())
            return max(0, min(1, score))
        except:
            gt_embedding = self.get_embedding(ground_truth)
            ans_embedding = self.get_embedding(answer)
            return self.cosine_similarity(gt_embedding, ans_embedding)
    
    def response_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""Evaluate how relevant the answer is to the question.
        
        Question: {question}
        Answer: {answer}
        
        Score the relevancy on a scale from 0 to 1, where 1 is perfectly relevant and 0 is completely irrelevant.
        Consider if the answer addresses the question completely without unnecessary information.
        
        Provide only the numerical score:"""
        
        response = self.query_llm(prompt)
        try:
            score = float(response.strip())
            return max(0, min(1, score))
        except:
      
            q_embedding = self.get_embedding(question)
            ans_embedding = self.get_embedding(answer)
            return self.cosine_similarity(q_embedding, ans_embedding)
    
    def factual_correctness(self, answer: str, context: str) -> float:
        prompt = f"""Evaluate how factually correct the answer is based on the provided context.
        
        Context: {context}
        Answer: {answer}
        
        Identify any claims in the answer and check if they are supported by the context.
        Score the factual correctness on a scale from 0 to 1, where 1 is completely factual and 0 has no factual basis.
        
        Provide only the numerical score:"""
        
        response = self.query_llm(prompt)
        try:
            score = float(response.strip())
            return max(0, min(1, score))
        except:
      
            answer_lower = answer.lower()
            context_lower = context.lower()
            if answer_lower in context_lower:
                return 0.8
            else:
               
                ans_embedding = self.get_embedding(answer)
                ctx_embedding = self.get_embedding(context)
                return self.cosine_similarity(ans_embedding, ctx_embedding) * 0.8
    
    def context_recall(self, ground_truth: str, retrieved_context: str) -> float:
        gt_embedding = self.get_embedding(ground_truth)
        ctx_embedding = self.get_embedding(retrieved_context)
        return self.cosine_similarity(gt_embedding, ctx_embedding)
    
    def evaluate_rag(self, questions: List[str], ground_truths: List[str], 
                    passages: List[str], documents: List[str]) -> Dict[str, float]:
        results = {
            "answer_correctness": [],
            "response_relevancy": [],
            "factual_correctness": [],
            "context_recall": []
        }
        
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
            question_embedding = self.get_embedding(question)
            passage_embeddings = [self.get_embedding(p) for p in passages]
            sim_scores = [self.cosine_similarity(question_embedding, emb) for emb in passage_embeddings]

            top_indices = np.argsort(sim_scores)[-3:][::-1]
            context = " ".join([passages[i] for i in top_indices])

            prompt = f"""Answer the question based only on the following context:
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            answer = self.query_llm(prompt)

            ac = self.answer_correctness(question, ground_truth, answer)
            rr = self.response_relevancy(question, answer)
            fc = self.factual_correctness(answer, context)
            cr = self.context_recall(ground_truth, context)
            
            results["answer_correctness"].append(ac)
            results["response_relevancy"].append(rr)
            results["factual_correctness"].append(fc)
            results["context_recall"].append(cr)

        return {k: np.mean(v) for k, v in results.items()}