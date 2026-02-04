import json
import os
from typing import List, Dict, Any
from chunking_methods import ChunkingMethods
from hope_core import HOPEMetric
from rag_evaluation import RAGEvaluator
import numpy as np
from read_doc import read_documents


class HOPEExperiment:
    def __init__(self):
        self.chunker = ChunkingMethods()
        self.hope_metric = HOPEMetric()
        self.rag_evaluator = RAGEvaluator()
    
    def load_documents(self, data_dir: str) -> List[Dict[str, Any]]:

        documents = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'id': filename.replace('.txt', ''),
                        'content': content,
                        'source': filename
                    })
        return documents
    
    def generate_questions(self, document: str, num_questions: int = 5) -> List[str]:
        """Generate questions for a document"""
        prompt = f"""Generate {num_questions} questions that can be answered from the following text:
        
        {document[:1000]}...
        
        Questions:
        1."""
        
        response = self.hope_metric.query_llm(prompt)
        questions = self.hope_metric._parse_llm_output(response, num_questions)
        return questions
    
    def run_experiment(self, documents: List[Dict[str, Any]], output_file: str):

        results = []
        
        chunking_strategies = [
            ('fixed_small', lambda text: self.chunker.fixed_size_chunking(text, 500, 50)),
            ('fixed_large', lambda text: self.chunker.fixed_size_chunking(text, 2000, 200)),
            ('recursive_small', lambda text: self.chunker.recursive_chunking(text, 500)),
            ('recursive_large', lambda text: self.chunker.recursive_chunking(text, 2000)),
            ('semantic', lambda text: self.chunker.semantic_chunking(text))
        ]
        
        for doc in documents:
            print(f"Processing document: {doc['id']}")
            

            questions = self.generate_questions(doc['content'], 5)
            ground_truths = questions
            
            for strategy_name, chunking_func in chunking_strategies:
                print(f"  Testing strategy: {strategy_name}")
                
                try:
                    passages = chunking_func(doc['content'])
                except Exception as e:
                    print(f"Error in chunking: {e}")
                    continue
 
                hope_results = self.hope_metric.calculate_hope(
                    doc['content'],
                    passages,
                    images=doc.get("images", [])
                )

   
                rag_results = self.rag_evaluator.evaluate_rag(
                    questions, ground_truths, passages, [doc['content']]
                )
             
                result = {
                    'document_id': doc['id'],
                    'chunking_strategy': strategy_name,
                    'num_passages': len(passages),
                    'hope_metrics': hope_results,
                    'rag_metrics': rag_results
                }
                
                results.append(result)
                print(f"    HOPE Score: {hope_results['hope_score']:.3f}")
                print(f"    RAG Accuracy: {rag_results['answer_correctness']:.3f}")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]):
  
        hope_scores = [r['hope_metrics']['hope_score'] for r in results]
        rag_scores = [r['rag_metrics']['answer_correctness'] for r in results]
        
        print(f"Average HOPE score: {np.mean(hope_scores):.3f}")
        print(f"Average RAG accuracy: {np.mean(rag_scores):.3f}")
        
        correlation = np.corrcoef(hope_scores, rag_scores)[0, 1]
        print(f"Correlation between HOPE and RAG accuracy: {correlation:.3f}")

def main():
    experiment = HOPEExperiment()

    data_dir = "documents"          
    output_file = "hope_results.json"

    os.makedirs(data_dir, exist_ok=True)


    documents = read_documents(data_dir)
    # print(documents)
    results = experiment.run_experiment(documents, output_file)

    experiment.analyze_results(results)

if __name__ == "__main__":
    main()