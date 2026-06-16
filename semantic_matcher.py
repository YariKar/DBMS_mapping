import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 3. АЛГОРИТМ ИЗВЛЕЧЕНИЯ ОТНОШЕНИЙ ---

class SemanticMatcher:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _clean(self, text):
        if not text: return ""
        doc = self.nlp(text.lower())
        return " ".join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']])

    def find_matches(self, df, threshold=0.55):
        df['clean_desc'] = df['description'].apply(self._clean)
        # Исключаем поля без описаний
        df = df[df['clean_desc'] != ""].reset_index(drop=True)
        
        embeddings = self.embedder.encode(df['clean_desc'].tolist(), show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        
        results = []
        n = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                # Исключаем сравнение поля с самим собой внутри одного контейнера
                if df.iloc[i]['system'] == df.iloc[j]['system'] and df.iloc[i]['container'] == df.iloc[j]['container']:
                    continue
                
                score = sim_matrix[i][j]
                if score >= threshold:
                    results.append({
                        "A_Sys": df.iloc[i]['system'], "A_Loc": f"{df.iloc[i]['container']}.{df.iloc[i]['element']}",
                        "B_Sys": df.iloc[j]['system'], "B_Loc": f"{df.iloc[j]['container']}.{df.iloc[j]['element']}",
                        "Score": round(float(score), 4),
                        "A_Desc": df.iloc[i]['description'], "B_Desc": df.iloc[j]['description']
                    })
        return pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
