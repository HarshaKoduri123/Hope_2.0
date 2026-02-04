import ollama
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
import re
from PIL import Image

def extract_facts(document):
        sentences = re.split(r'(?<=\.|\?)\s+', document)
        return [s for s in sentences if len(s.split()) >= 6]

class HOPEMetric:
    def __init__(self, model_name="llama3"):
        self.llm_model = model_name
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.clip_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32'
        )

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.clip_model.encode(image, normalize_embeddings=True)

    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.embedding_model.encode(text, normalize_embeddings=True)
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(vec_a, vec_b))
    
    def query_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Query local LLM using Ollama"""
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
    
    def calculate_zeta_con(
        self,
        passage: str,
        images: List[Dict[str, Any]] = None,
        num_statements: int = 5
    ) -> float:
        """
        ζ_con⁺ : Textual + Visual Concept Unity
        """

        # ---- Textual concept unity (unchanged, solid) ----
        prompt = f"""Generate {num_statements} distinct factual statements about the following text.
        The statements should capture its core concepts and details:

        Text: {passage}

        Statements:
        1."""
        
        llm_output = self.query_llm(prompt, temperature=0.7)
        statements = self._parse_llm_output(llm_output, num_statements)

        if len(statements) < 2:
            return 0.5

        embeddings = [self.get_embedding(s) for s in statements]

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(self.cosine_similarity(embeddings[i], embeddings[j]))

        zeta_con_text = float(np.clip(np.mean(sims), 0.0, 1.0))

        if images:
            text_emb = self.clip_model.encode(passage, normalize_embeddings=True)

            image_scores = []
            for img in images:
                img_emb = self.get_image_embedding(img["image_path"])
                image_scores.append(max(0.0, np.dot(text_emb, img_emb)))

            zeta_visual = float(np.clip(max(image_scores), 0.0, 1.0))

            return 0.7 * zeta_con_text + 0.3 * zeta_visual

        return zeta_con_text



    
    def calculate_zeta_sem(self, passages: List[str]) -> float:
        """
        Semantic Independence (ζ_sem)
        Measures maximum cross-passage semantic leakage.
        """
        if len(passages) < 2:
            return 1.0 

        embeddings = [self.get_embedding(p) for p in passages]
        max_similarities = []

        for i in range(len(embeddings)):
            sims = []
            for j in range(len(embeddings)):
                if i != j:
                    sims.append(np.dot(embeddings[i], embeddings[j]))
            max_similarities.append(max(sims))

        zeta_sem = 1.0 - np.mean(max_similarities)
        return float(np.clip(zeta_sem, 0.0, 1.0))


    

    def calculate_zeta_inf(
        self,
        document: str,
        passages: List[str],
        k: int = 3,
        tau: float = 0.6
    ) -> float:
        """
        Information Preservation (ζ_inf)
        Fraction of document facts recoverable via passage retrieval.
        """
        facts = extract_facts(document)
        if not facts or not passages:
            return 0.5

        passage_embeddings = [self.get_embedding(p) for p in passages]
        recovered = 0

        for fact in facts:
            f_emb = self.get_embedding(fact)
            sims = [np.dot(f_emb, p_emb) for p_emb in passage_embeddings]
            top_k = sorted(sims, reverse=True)[:k]

            if max(top_k) >= tau:
                recovered += 1

        return recovered / len(facts)
    
    def calculate_zeta_align(
        self,
        passages: List[str],
        images: List[Dict[str, Any]],
        window: int = 1
    ) -> float:
        """
        ζ_align: Measures how well images align with nearby text chunks
        """

        if not images or not passages:
            return 1.0 

        text_embeddings = [
            self.clip_model.encode(p, normalize_embeddings=True)
            for p in passages
        ]

        scores = []

        for img in images:
            img_emb = self.get_image_embedding(img["image_path"])
            sims = [np.dot(img_emb, t_emb) for t_emb in text_embeddings]
            scores.append(max(0.0, max(sims)))


        return float(np.clip(np.mean(scores), 0.0, 1.0))




    
    def _parse_llm_output(self, output: str, expected_items: int) -> List[str]:
        items = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
          
            if re.match(r'^\d+[\.\)]', line):
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
            
            if line and not line.startswith(('Statements:', 'Questions:')):
                items.append(line)
                
                if len(items) >= expected_items:
                    break
        
        return items
    
    def _parse_statements(self, output: str) -> Dict[str, List[str]]:
        result = {'true': None, 'false': []}
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('true:'):
                result['true'] = line[5:].strip()
            elif line.lower().startswith('false1:'):
                result['false'].append(line[7:].strip())
            elif line.lower().startswith('false2:'):
                result['false'].append(line[7:].strip())
            elif line.lower().startswith('false3:'):
                result['false'].append(line[7:].strip())
        
        return result
    
    def calculate_hope(self, document, passages, images=None):

        print(f"Calculating HOPE for {len(passages)} passages...")

        zeta_con_values = [
            self.calculate_zeta_con(p, images=images)
            for p in passages
        ]

        zeta_con = np.mean(zeta_con_values) if zeta_con_values else 0.5


        zeta_sem = self.calculate_zeta_sem(passages)

  
        zeta_inf = self.calculate_zeta_inf(document, passages)

        zeta_align = self.calculate_zeta_align(passages, images)


        hope_score = (zeta_con + zeta_sem + zeta_inf + zeta_align) / 4.0


        return {
            "hope_score": hope_score,
            "zeta_con": zeta_con,
            "zeta_sem": zeta_sem,
            "zeta_inf": zeta_inf,
            "zeta_align": zeta_align,
            "zeta_con_values": zeta_con_values
        }

