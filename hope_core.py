import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import re
from PIL import Image
import random


class HOPEMetric:
    def __init__(self, model_name="llama3"):
        self.llm_model = model_name

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.clip_model = SentenceTransformer(
            "sentence-transformers/clip-ViT-B-32"
        )

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.clip_model.encode(image, normalize_embeddings=True)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))



    def query_llm(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return ""

    # --------------------------------------------------
    # ζ_con — Concept Unity
    # --------------------------------------------------

    def calculate_zeta_con(self, passage: str, num_statements: int = 5) -> float:
        prompt = f"""
        Generate {num_statements} distinct factual statements about the following text.
        The statements should capture its core concepts and details.

        Text: {passage}

        Statements:
        1.
        """

        llm_output = self.query_llm(prompt)
        statements = self._parse_llm_output(llm_output, num_statements)

        if len(statements) < 2:
            return 0.5

        embeddings = [self.get_embedding(s) for s in statements]

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(self.cosine_similarity(embeddings[i], embeddings[j]))

        return float(max(0.0, np.mean(sims)))

    # --------------------------------------------------
    # ζ_sem — Semantic Independence
    # --------------------------------------------------

    def calculate_zeta_sem(
        self,
        focus_passage: str,
        other_passages: List[str],
        num_questions: int = 3
    ) -> float:

        prompt = f"""
        Based on the following text, generate {num_questions} questions that can be
        answered using ONLY the information contained within it.

        Text: {focus_passage}

        Questions:
        1.
        """

        llm_output = self.query_llm(prompt)
        questions = self._parse_llm_output(llm_output, num_questions)

        if not questions:
            return 0.5

        similarities = []

        for q in questions:
            focus_prompt = f"""
            Answer the question using ONLY the context below.

            Context: {focus_passage}

            Question: {q}
            Answer:
            """
            ans_focus = self.query_llm(focus_prompt, temperature=0.1)
            context = focus_passage + "\n\n" + "\n\n".join(other_passages)
            context_prompt = f"""
            Answer the question using the context below.

            Context: {context}

            Question: {q}
            Answer:
            """
            ans_context = self.query_llm(context_prompt, temperature=0.1)

            if ans_focus and ans_context:
                emb_a = self.get_embedding(ans_focus)
                emb_b = self.get_embedding(ans_context)
                similarities.append(self.cosine_similarity(emb_a, emb_b))

        return float(max(0.0, np.mean(similarities))) if similarities else 0.5

    # --------------------------------------------------
    # ζ_inf — Information Preservation
    # --------------------------------------------------

    def calculate_zeta_inf(
        self,
        document: str,
        passages: List[str],
        num_samples: int = 5
    ) -> float:
        if not passages or len(document) < 100:
            return 0.5

        sentences = []
        for sent in re.split(r'(?<=[.!?])\s+', document):
            if sent.strip() and len(sent.split()) > 3:
                sentences.append(sent.strip())
        
        if len(sentences) < 3:
            return 0.5

        passage_embeddings = self.embedding_model.encode(
            passages,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        correct = 0
        total_tests = 0
        max_samples = min(num_samples, len(sentences) - 2)
        
        indices = random.sample(range(len(sentences) - 2), max_samples)
        
        for i in indices:
            segment = " ".join(sentences[i:i + 3])
            segment_embedding = self.embedding_model.encode(
                [segment],
                normalize_embeddings=True
            )[0]
            
            similarities = np.dot(passage_embeddings, segment_embedding)
            best_idx = np.argmax(similarities)
            context = passages[best_idx] 
            
            gen_prompt = f"""Generate 4 statements about this text:

                "{segment}"

                Requirements:
                • ONE statement must be TRUE and directly supported by the text
                • THREE statements must be FALSE but plausible
                • All statements should be concise (1-2 sentences)

                Format:
                True: [true statement]
                False1: [false statement]
                False2: [false statement]
                False3: [false statement]"""
                            
            llm_output = self.query_llm(gen_prompt, temperature=0.7)
            stmts = self._parse_statements(llm_output)
            
            if not stmts["true"] or len(stmts["false"]) < 3:
                continue

            all_statements = [stmts["true"]] + stmts["false"]
            random.shuffle(all_statements)
            true_position = all_statements.index(stmts["true"]) + 1

            test_prompt = f"""Based on the following text, identify which statement is TRUE:

                TEXT:
                {context}

                STATEMENTS:
                1. {all_statements[0]}
                2. {all_statements[1]}
                3. {all_statements[2]}
                4. {all_statements[3]}

                Which statement number (1, 2, 3, or 4) is TRUE according to the text above?
                Answer with just the number:"""
            
            answer = self.query_llm(test_prompt, temperature=0.1)
            answer = answer.strip()
            numbers = re.findall(r'\d+', answer)
            if numbers:
                chosen = int(numbers[0])
                if 1 <= chosen <= 4:
                    if chosen == true_position:
                        correct += 1
                        print(f"Correct: {chosen} (true was {true_position})")
                    else:
                        print(f"Wrong: {chosen} (true was {true_position})")
                else:
                    print(f"Invalid number: {chosen}")
            else:
                print(f"No number in answer: '{answer}'")
            
            total_tests += 1
        
        if total_tests == 0:
            return 0.5
        
        score = correct / total_tests
        print(f"ζ_inf Score: {correct}/{total_tests} = {score:.3f}")
        return score


    # --------------------------------------------------
    # ζ_align — Image–Text Alignment
    # --------------------------------------------------

    def calculate_zeta_align(
        self,
        passages: List[str],
        images: List[Dict[str, Any]]
    ) -> float:

        if not images or not passages:
            return 1.0

        text_embs = self.clip_model.encode(
            passages,
            normalize_embeddings=True
        )

        image_embs = []
        for img in images:
            image_embs.append(
                self.get_image_embedding(img["image_path"])
            )
        image_embs = np.array(image_embs)
        sim_matrix = image_embs @ text_embs.T
        best_scores = np.max(sim_matrix, axis=1)
        k = min(2, len(best_scores))
        topk_scores = np.sort(best_scores)[-k:]

        return float(np.clip(np.mean(topk_scores), 0.0, 1.0))

    def calculate_hope(
        self,
        document: str,
        passages: List[str],
        images: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:


        print(f"Calculating HOPE for {len(passages)} passages...")
        print(f"Calculating ζ_inf (Information Preservation)...")
        zeta_inf = self.calculate_zeta_inf(document, passages)
        print(f"ζ_inf = {zeta_inf:.3f}")
        
        print(f"Calculating ζ_con (Concept Unity)...")
        zeta_con_vals = [self.calculate_zeta_con(p) for p in passages]
        zeta_con = float(np.mean(zeta_con_vals))
        print(f"ζ_con = {zeta_con:.3f}")
        
        print(f"Calculating ζ_sem (Semantic Independence)...")
        zeta_sem_vals = []
        for i, p in enumerate(passages):
            others = passages[:i] + passages[i + 1:]
            zeta_sem_val = self.calculate_zeta_sem(p, others)
            zeta_sem_vals.append(zeta_sem_val)
        zeta_sem = float(np.mean(zeta_sem_vals))
        print(f"ζ_sem = {zeta_sem:.3f}")
        
        print(f"Calculating ζ_align (Multimodal Alignment)...")
        zeta_align = self.calculate_zeta_align(passages, images) if images else 1.0
        print(f"ζ_align = {zeta_align:.3f}")
        
        hope_score = (zeta_con + zeta_sem + zeta_inf + zeta_align) / 4.0
        
        print(f"FINAL HOPE SCORE: {hope_score:.3f}")
        print(f"ζ_con: {zeta_con:.3f}")
        print(f"ζ_sem: {zeta_sem:.3f}")
        print(f"ζ_inf: {zeta_inf:.3f}")
        print(f"ζ_align: {zeta_align:.3f}")
        
        return {
            "hope_score": hope_score,
            "zeta_con": zeta_con,
            "zeta_sem": zeta_sem,
            "zeta_inf": zeta_inf,
            "zeta_align": zeta_align,

        }



    def _parse_llm_output(self, output: str, expected: int) -> List[str]:
        items = []
        for line in output.split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if line:
                items.append(line)
            if len(items) >= expected:
                break
        return items

    def _parse_statements(self, output: str) -> Dict[str, List[str]]:
        result = {"true": None, "false": []}
        output = output.strip()
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('true:'):
                result["true"] = line[5:].strip()
            elif re.match(r'^false\s*\d*:', line.lower()):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    result["false"].append(parts[1].strip())

        if not result["true"] and len(result["false"]) < 3:
            for i, line in enumerate(lines):
                if line.strip() and re.match(r'^\d+[\.\)]', line.strip()):
                    content = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if i == 0 and not result["true"]:
                        result["true"] = content
                    else:
                        result["false"].append(content)
        
        if result["false"] and len(result["false"]) > 3:
            result["false"] = result["false"][:3]
        
        return result