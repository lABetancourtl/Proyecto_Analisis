# requerimiento3/req3_palabras.py
import streamlit as st
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- CARGA DE CATEGORÍAS Y SINÓNIMOS ---
@st.cache_data
def cargar_categorias(categorias_json_path):
    with open(categorias_json_path, 'r', encoding='utf-8') as f:
        categorias_data = json.load(f)

    categorias = {}
    sinonimos = {}

    for categoria in categorias_data['Categorías']:
        cat_name = categoria['nombre']
        categorias[cat_name] = []
        for variable in categoria['variables']:
            vars_split = [v.strip() for v in variable.split('-') if v.strip()]
            primary_var = vars_split[0]
            categorias[cat_name].append(primary_var)
            for synonym in vars_split[1:]:
                sinonimos[synonym.lower()] = primary_var.lower()

    return categorias, sinonimos

# --- FUNCIONES AUXILIARES ---
def normalize_text(text):
    """Normaliza texto: minúsculas y quita signos de puntuación"""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    return text

def parse_ris_file(filepath):
    """Parsea archivo RIS y devuelve lista de abstracts"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    records = []
    current_record = {}

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('AB  - '):
            current_record['abstract'] = line[6:]
        elif line.startswith('ER  -'):
            if current_record and 'abstract' in current_record:
                records.append(current_record)
            current_record = {}
    return records

def find_terms_in_abstract(abstract, categoria, categorias, sinonimos):
    abstract_norm = normalize_text(abstract)
    terms_found = set()
    for term in categorias[categoria]:
        term_norm = term.lower()
        if re.search(r'\b' + re.escape(term_norm) + r'\b', abstract_norm):
            terms_found.add(term)
        for syn, primary in sinonimos.items():
            if primary == term_norm and re.search(r'\b' + re.escape(syn) + r'\b', abstract_norm):
                terms_found.add(term)
    return terms_found

# --- FUNCIÓN PRINCIPAL PARA STREAMLIT ---
def mostrar_requerimiento_3(project_root):
    st.markdown('<div class="requirement-title">Requerimiento 3: Palabras Clave y Frecuencia</div>', unsafe_allow_html=True)

    # --- Cargar categorías ---
    project_root = Path(project_root)  # Convierte a Path
    categorias_json_path = project_root / 'requerimiento3' / 'categorias.json'
    categorias, sinonimos = cargar_categorias(categorias_json_path)

    # --- Selección de categoría ---
    categorias_list = list(categorias.keys())
    
    # Fijar directamente la categoría
    selected_categoria = "Concepts of Generative AI in Education"
    st.write(f"Analizando categoría: **{selected_categoria}**")

    # --- Parsear archivo RIS ---
    ris_file = project_root / 'resultados' / 'requerimiento1' / 'resultados_unificados.ris'
    if not ris_file.exists():
        st.warning("⚠️ No se encontró el archivo unificado de Requerimiento 1. Ejecuta primero R1.")
        return

    records = parse_ris_file(ris_file)

    # --- Contar frecuencia ---
    freq_global = Counter()
    freq_categoria = Counter()
    cooccurrence_matrix = Counter()

    for record in records:
        abstract = record.get('abstract', '')
        if not abstract:
            continue
        terms = find_terms_in_abstract(abstract, selected_categoria, categorias, sinonimos)
        freq_categoria.update(terms)
        words_in_abstract = normalize_text(abstract).split()
        freq_global.update(words_in_abstract)
        # Co-ocurrencia
        for pair in combinations(sorted(terms), 2):
            cooccurrence_matrix[pair] += 1

    # --- Mostrar tabla de frecuencia ---
    st.subheader(f"Frecuencia de términos - {selected_categoria}")
    import pandas as pd
    if freq_categoria:
        df_freq = pd.DataFrame(freq_categoria.items(), columns=['Término', 'Frecuencia']).sort_values(by='Frecuencia', ascending=False)
        st.table(df_freq)
    else:
        st.info("No se encontraron términos de esta categoría en los abstracts.")

    # --- Nube de palabras ---
    st.subheader("Nube de palabras")
    if freq_categoria:
        plt.figure(figsize=(10,5))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_categoria)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()
    else:
        st.info("No hay términos para generar nube de palabras.")

    # --- Sugerencia de nuevas palabras ---
    st.subheader("Palabras nuevas sugeridas (Top 15)")
    top_new_words = [w for w, f in freq_global.most_common(15)
                    if w.lower() not in [t.lower() for t in categorias[selected_categoria]]]
    st.write(top_new_words)

    # --- Precisión estimada de las palabras nuevas ---
    palabras_existentes = [t.lower() for t in categorias[selected_categoria]]
    nuevas = [w.lower() for w in top_new_words]
    n_correctas = sum(1 for w in nuevas if w in palabras_existentes)  # esto compara con la lista original
    precision = n_correctas / len(nuevas) if nuevas else 0
    st.write(f"Precisión estimada de las palabras nuevas: {precision:.2f}")


    # --- Opcional: Red de co-ocurrencia ---
    import networkx as nx
    if cooccurrence_matrix:
        st.subheader("Red de co-ocurrencia")
        G = nx.Graph()
        for term in freq_categoria.keys():
            G.add_node(term)
        for (t1, t2), w in cooccurrence_matrix.items():
            G.add_edge(t1, t2, weight=w)
        plt.figure(figsize=(12,8))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_size=50)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title('Red de Co-ocurrencia de Términos', fontsize=14)
        plt.axis('off')
        st.pyplot(plt)
        plt.close()
