# requerimiento2_similitud.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein  # rápido
import jellyfish
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any

# ------------------------------
# Helpers: cargar RIS simple
# ------------------------------
def load_unified_ris(ris_path: str) -> pd.DataFrame:
    """
    Carga registros del RIS unificado y devuelve DataFrame con al menos columnas:
    'TI' (title) y 'AB' (abstract). Si no hay AB pondrá ''.
    """
    entries = []
    current = {}
    with open(ris_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('TY  -'):
                current = {'TI': '', 'AB': ''}
            elif line.startswith('ER  -'):
                entries.append(current)
                current = {}
            elif '  - ' in line:
                tag, val = line.split('  - ', 1)
                tag = tag.strip()
                if tag == 'TI':
                    current['TI'] = current.get('TI', '') + val.strip()
                elif tag in ('AB', 'N2'):
                    current['AB'] = current.get('AB', '') + val.strip()
                else:
                    # other tags ignored for now
                    pass
    df = pd.DataFrame(entries)
    if 'TI' not in df.columns:
        df['TI'] = ''
    if 'AB' not in df.columns:
        df['AB'] = ''
    # fill NA
    df['AB'] = df['AB'].fillna('')
    df['TI'] = df['TI'].fillna('')
    return df

# ------------------------------
# Preprocesamiento simple
# ------------------------------
import re
import unicodedata
def normalize_text(s: str) -> str:
    if not s:
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# ------------------------------
# Algoritmos de similitud
# Cada función devuelve: similarity (0..1) y 'steps' dict de intermedios para explicación
# ------------------------------

def levenshtein_similarity(a: str, b: str) -> Dict[str, Any]:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n and not b_n:
        return {'score': 1.0, 'steps': {'lev': 0, 'len_max':0}}
    # distancia
    dist = Levenshtein.distance(a_n, b_n)
    max_len = max(len(a_n), len(b_n)) or 1
    score = 1.0 - (dist / max_len)
    return {'score': float(score), 'steps': {'lev': int(dist), 'len_max': int(max_len)}}

def jaro_winkler_similarity(a: str, b: str) -> Dict[str, Any]:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n and not b_n:
        return {'score': 1.0, 'steps': {}}
    jw = jellyfish.jaro_winkler_similarity(a_n, b_n)
    return {'score': float(jw), 'steps': {}}

def jaccard_shingle_similarity(a: str, b: str, k: int = 3) -> Dict[str, Any]:
    a_n, b_n = normalize_text(a), normalize_text(b)
    def shingles(s):
        s2 = re.sub(r'\s+', ' ', s)
        tokens = s2.split()
        if len(tokens) == 0:
            return set()
        # build k-shingles on tokens
        sh = set()
        for i in range(max(1, len(tokens)-k+1)):
            sh.add(' '.join(tokens[i:i+k]))
        return sh
    sa, sb = shingles(a_n), shingles(b_n)
    inter = sa.intersection(sb)
    union = sa.union(sb)
    j = 0.0
    if union:
        j = len(inter) / len(union)
    return {'score': float(j), 'steps': {'|A∩B|': len(inter), '|A∪B|': len(union), 'k': k}}


def cosine_tfidf_similarity(a_list):
    """
    Calcula la similitud coseno usando TF-IDF entre una lista de textos.
    Maneja textos vacíos automáticamente.
    """
    # Normalizar y filtrar textos vacíos
    normed = [t.lower().strip() for t in a_list if t and t.strip() != '']
    
    # Si después de filtrar quedan menos de 2 textos, duplicamos el primero
    if len(normed) < 2:
        if len(normed) == 1:
            normed = normed * 2
        else:
            # Ningún texto válido
            normed = ["", ""]

    try:
        # vectorizador sin stop_words para evitar problemas de idioma
        vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
        X = vectorizer.fit_transform(normed)
        sim_matrix = cosine_similarity(X)
        feature_names = vectorizer.get_feature_names_out()[:20].tolist()
        return {'matrix': sim_matrix, 'steps': {'n_features': X.shape[1], 'sample_features': feature_names}}
    except ValueError:
        # fallback: similitud identidad si TF-IDF falla
        n = len(normed)
        sim_matrix = np.eye(n)
        return {'matrix': sim_matrix, 'steps': {'n_features': 0, 'sample_features': []}}

# SBERT (sentence-transformers)
_sbert_model = None
def get_sbert_model(name='all-MiniLM-L6-v2'):
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(name)
    return _sbert_model

def sbert_similarity(a_list: List[str], model_name='all-MiniLM-L6-v2') -> Dict[str, Any]:
    model = get_sbert_model(model_name)
    texts = [normalize_text(t) for t in a_list]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    sim = cosine_similarity(embeddings)
    return {'matrix': sim, 'steps': {'embedding_dim': int(embeddings.shape[1])}}

# ------------------------------
# Orquestador: recibe indices seleccionados y realiza todos los métodos
# ------------------------------
def compute_similarities(texts: List[str], methods: List[str] = None) -> Dict[str, Any]:
    """
    texts: list of abstracts selected (order matters)
    methods: list of method keys: 'lev', 'jw', 'jaccard', 'tfidf', 'sbert'
    Returns a dict with scores by pair and matrices for tfidf/sbert
    """
    if methods is None:
        methods = ['lev', 'jw', 'jaccard', 'tfidf', 'sbert']

    n = len(texts)
    pairs = list(combinations(range(n), 2))
    result = {'pairs': [], 'matrices': {}, 'details': {}}

    # Pairwise classical per pair
    for i,j in pairs:
        a, b = texts[i], texts[j]
        pair_key = f"{i}-{j}"
        result['pairs'].append({'i': i, 'j': j, 'title': f"{i} <> {j}"})
        # compute requested
        det = {}
        if 'lev' in methods:
            det['levenshtein'] = levenshtein_similarity(a,b)
        if 'jw' in methods:
            det['jaro_winkler'] = jaro_winkler_similarity(a,b)
        if 'jaccard' in methods:
            det['jaccard'] = jaccard_shingle_similarity(a,b, k=3)
        result['details'][pair_key] = det

    # vector/matrix methods
    if 'tfidf' in methods:
        tf = cosine_tfidf_similarity(texts)
        result['matrices']['tfidf'] = tf['matrix'].tolist()
        result['details']['tfidf_steps'] = tf['steps']
    if 'sbert' in methods:
        sb = sbert_similarity(texts)
        result['matrices']['sbert'] = sb['matrix'].tolist()
        result['details']['sbert_steps'] = sb['steps']

    return result

# ------------------------------
# Save results helper
# ------------------------------
def save_sim_results(output_dir: str, data: dict):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'similitud_resultados.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return out_path
