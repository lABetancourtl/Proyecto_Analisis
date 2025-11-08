import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClusteringAnalyzer:
    def __init__(self, project_root):
        self.project_root = project_root
        self.df = None
        self.abstracts = []
        self.titles = []
        self.results = {}
        
    def load_data(self):
        """Carga los datos del archivo unificado del Requerimiento 1"""
        unified_path = os.path.join(self.project_root, "resultados", "requerimiento1", "resultados_unificados.ris")
        
        if not os.path.exists(unified_path):
            raise FileNotFoundError("No se encontró el archivo unificado. Ejecuta primero el Requerimiento 1.")
        
        # Usar la misma función de carga del Requerimiento 2
        from requerimiento2.requerimiento2_similitud import load_unified_ris
        self.df = load_unified_ris(unified_path)
        
        # Filtrar abstracts no vacíos
        valid_indices = []
        self.abstracts = []
        self.titles = []
        
        for idx, row in self.df.iterrows():
            abstract = row['AB'] if pd.notna(row['AB']) else row['TI']
            title = row['TI'] if pd.notna(row['TI']) else f"Documento {idx}"
            
            if abstract and str(abstract).strip() and len(str(abstract).strip()) > 50:
                self.abstracts.append(str(abstract).strip())
                self.titles.append(title)
                valid_indices.append(idx)
        
        if len(self.abstracts) < 3:
            raise ValueError("Se necesitan al menos 3 documentos con abstracts válidos para clustering")
        
        st.info(f"Cargados {len(self.abstracts)} documentos con abstracts válidos")
        return len(self.abstracts)
    
    def preprocess_text(self, texts):
        """Preprocesamiento de texto para clustering"""
        import re
        import unicodedata
        
        def clean_text(text):
            # Normalizar y limpiar texto
            text = str(text).lower()
            text = unicodedata.normalize('NFKD', text)
            text = re.sub(r'[^\w\s]', ' ', text)  # Remover puntuación
            text = re.sub(r'\d+', ' ', text)      # Remover números
            text = re.sub(r'\s+', ' ', text)      # Remover espacios múltiples
            return text.strip()
        
        return [clean_text(text) for text in texts]
    
    def create_tfidf_embeddings(self, texts):
        """Crea embeddings TF-IDF"""
        processed_texts = self.preprocess_text(texts)
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        return tfidf_matrix.toarray(), vectorizer
    
    def create_sbert_embeddings(self, texts):
        """Crea embeddings usando SBERT"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def compute_distance_matrix(self, embeddings, method='cosine'):
        """Calcula matriz de distancias"""
        if method == 'cosine':
            similarity_matrix = cosine_similarity(embeddings)
            distance_matrix = 1 - similarity_matrix
        elif method == 'euclidean':
            distance_matrix = pdist(embeddings, metric='euclidean')
            distance_matrix = squareform(distance_matrix)
        else:
            raise ValueError("Método de distancia no soportado")
        
        return distance_matrix
    
    def hierarchical_clustering(self, distance_matrix, method='ward'):
        """Aplica clustering jerárquico"""
        # Convertir a forma condensada para scipy
        condensed_dist = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
        
        # Aplicar linkage
        Z = linkage(condensed_dist, method=method)
        return Z
    
    def calculate_clustering_metrics(self, embeddings, labels):
        """Calcula métricas de calidad del clustering"""
        if len(np.unique(labels)) < 2:
            return {
                'silhouette': -1,
                'calinski_harabasz': -1,
                'davies_bouldin': float('inf')
            }
        
        try:
            silhouette = silhouette_score(embeddings, labels)
            calinski = calinski_harabasz_score(embeddings, labels)
            davies = davies_bouldin_score(embeddings, labels)
        except:
            silhouette = -1
            calinski = -1
            davies = float('inf')
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies
        }
    
    def plot_dendrogram(self, Z, titles, method_name, threshold=None):
        """Genera y muestra dendrograma"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Crear dendrograma
        dendrogram(
            Z,
            labels=titles,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True,
            ax=ax
        )
        
        if threshold:
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Umbral: {threshold:.2f}')
        
        plt.title(f'Dendrograma - {method_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Documentos')
        plt.ylabel('Distancia')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def analyze_clustering(self, max_docs=50):
        """Análisis principal de clustering"""
        # Limitar número de documentos para mejor visualización
        if len(self.abstracts) > max_docs:
            st.warning(f"Limite de documentos({len(self.abstracts)}). Se han definido {max_docs} documentos para analizar.")
            texts = self.abstracts[:max_docs]
            titles = [f"Doc {i+1}: {self.titles[i][:50]}..." for i in range(max_docs)]
        else:
            texts = self.abstracts
            titles = [f"Doc {i+1}: {title[:50]}..." for i, title in enumerate(self.titles)]
        
        # Métodos de clustering a comparar
        linkage_methods = ['ward', 'complete', 'average']
        embedding_methods = ['tfidf', 'sbert']
        
        results = {}
        
        for emb_method in embedding_methods:
            st.subheader(f"Embeddings: {emb_method.upper()}")
            
            # Crear embeddings
            if emb_method == 'tfidf':
                embeddings, vectorizer = self.create_tfidf_embeddings(texts)
            else:  # sbert
                embeddings = self.create_sbert_embeddings(texts)
            
            # Matriz de distancias
            distance_matrix = self.compute_distance_matrix(embeddings, 'cosine')
            
            for link_method in linkage_methods:
                st.write(f"**Método de linkage:** {link_method}")
                
                try:
                    # Clustering jerárquico
                    Z = self.hierarchical_clustering(distance_matrix, link_method)
                    
                    # Determinar umbral automático (percentil 70 de las distancias)
                    threshold = np.percentile(Z[:, 2], 70)
                    
                    # Asignar clusters
                    clusters = fcluster(Z, threshold, criterion='distance')
                    
                    # Calcular métricas
                    metrics = self.calculate_clustering_metrics(embeddings, clusters)
                    
                    # Guardar resultados
                    key = f"{emb_method}_{link_method}"
                    results[key] = {
                        'linkage_matrix': Z.tolist(),
                        'clusters': clusters.tolist(),
                        'threshold': threshold,
                        'metrics': metrics,
                        'n_clusters': len(np.unique(clusters))
                    }
                    
                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Silhouette", f"{metrics['silhouette']:.3f}")
                    with col2:
                        st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.3f}")
                    with col3:
                        st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}")
                    
                    # Plot dendrogram
                    fig = self.plot_dendrogram(Z, titles, f"{emb_method.upper()} - {link_method}", threshold)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error con {emb_method}-{link_method}: {str(e)}")
        
        return results
    
    def determine_best_algorithm(self, results):
        """Determina el mejor algoritmo basado en métricas"""
        best_score = -float('inf')
        best_algorithm = None
        
        for algo, data in results.items():
            metrics = data['metrics']
            # Score compuesto (mayor es mejor)
            score = (metrics['silhouette'] + 
                    metrics['calinski_harabasz'] / 1000 -  # Normalizar
                    metrics['davies_bouldin'] / 10)
            
            if score > best_score:
                best_score = score
                best_algorithm = algo
        
        return best_algorithm, results[best_algorithm]
    
    def save_results(self, results, best_algorithm):
        """Guarda resultados del clustering"""
        output_dir = os.path.join(self.project_root, "resultados", "requerimiento4")
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar resultados en JSON
        results_data = {
            'best_algorithm': best_algorithm,
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_documents': len(self.abstracts)
        }
        
        json_path = os.path.join(output_dir, "clustering_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
        
        # Guardar figura del mejor algoritmo
        best_emb, best_link = best_algorithm.split('_')
        
        if len(self.abstracts) > 50:
            texts = self.abstracts[:50]
            titles = [f"Doc {i+1}" for i in range(50)]
        else:
            texts = self.abstracts
            titles = [f"Doc {i+1}: {title[:30]}..." for i, title in enumerate(self.titles)]
        
        # Recrear el mejor dendrograma
        if best_emb == 'tfidf':
            embeddings, _ = self.create_tfidf_embeddings(texts)
        else:
            embeddings = self.create_sbert_embeddings(texts)
        
        distance_matrix = self.compute_distance_matrix(embeddings, 'cosine')
        Z = self.hierarchical_clustering(distance_matrix, best_link)
        threshold = results[best_algorithm]['threshold']
        
        fig = self.plot_dendrogram(Z, titles, f"MEJOR: {best_emb.upper()} - {best_link}", threshold)
        fig_path = os.path.join(output_dir, "best_dendrogram.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        return json_path, fig_path

def mostrar_requerimiento_4(project_root):
    
    st.markdown('<div class="requirement-title">Requerimiento 4: Dendrograma de Similitud</div>', unsafe_allow_html=True)
    st.write("""
    Este requerimiento implementa tres algoritmos de agrupamiento jerárquico para construir dendrogramas 
    que representen la similitud entre abstracts científicos.
    """)
    
    # Inicializar analizador
    analyzer = HierarchicalClusteringAnalyzer(project_root)
    
    try:
        # Cargar datos
        n_docs = analyzer.load_data()
        
        st.subheader("Configuración del Análisis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Documentos cargados:** {n_docs}")
        with col2:
            max_docs = st.slider("Límite de documentos para análisis", 
                               min_value=10, max_value=100, value=min(50, n_docs))
        
        if st.button("Ejecutar Análisis de Clustering", type="primary"):
            with st.spinner("Realizando clustering jerárquico..."):
                # Ejecutar análisis
                results = analyzer.analyze_clustering(max_docs=max_docs)
                
                if results:
                    # Determinar mejor algoritmo
                    best_algo, best_data = analyzer.determine_best_algorithm(results)
                    
                    st.subheader("Mejor Algoritmo de Clustering")
                    st.success(f"**{best_algo.upper()}**")
                    
                    col1, col2, col3 = st.columns(3)
                    metrics = best_data['metrics']
                    with col1:
                        st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}",
                                help="Valores cercanos a 1 indican clusters bien definidos")
                    with col2:
                        st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.3f}",
                                help="Mayor valor indica clusters más densos y separados")
                    with col3:
                        st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}",
                                help="Menor valor indica mejor separación entre clusters")
                    
                    st.info(f"**Número de clusters encontrados:** {best_data['n_clusters']}")
                    
                    # Guardar resultados
                    json_path, fig_path = analyzer.save_results(results, best_algo)
                    
                    st.success("Análisis completado exitosamente")
                    st.write(f"**Resultados guardados en:**")
                    st.write(f"- JSON: `{json_path}`")
                    st.write(f"- Dendrograma: `{fig_path}`")
                    
                    # Mostrar explicación de coherencia
                    st.subheader("Análisis de Coherencia de Clusters")
                    
                    if metrics['silhouette'] > 0.5:
                        st.success("**Alta coherencia:** Los clusters están bien definidos y separados")
                    elif metrics['silhouette'] > 0.25:
                        st.warning("**Coherencia media:** Estructura de clusters razonable")
                    else:
                        st.error("**Baja coherencia:** Los clusters pueden superponerse significativamente")
                
                else:
                    st.error("No se pudieron generar resultados de clustering")
    
    except FileNotFoundError as e:
        st.error(f"{str(e)}")
    except ValueError as e:
        st.error(f"{str(e)}")
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")

# Función principal para ejecución independiente
def main():
    """Función principal para ejecutar el requerimiento 4 de forma independiente"""
    print("=== Clustering Jerárquico - Requerimiento 4 ===")
    
    # Obtener ruta del proyecto
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    
    analyzer = HierarchicalClusteringAnalyzer(PROJECT_ROOT)
    
    try:
        n_docs = analyzer.load_data()
        print(f"Cargados {n_docs} documentos")
        
        print("Ejecutando análisis de clustering...")
        results = analyzer.analyze_clustering()
        
        if results:
            best_algo, best_data = analyzer.determine_best_algorithm(results)
            print(f"Mejor algoritmo: {best_algo}")
            print(f"Métricas: {best_data['metrics']}")
            
            # Guardar resultados
            json_path, fig_path = analyzer.save_results(results, best_algo)
            print(f"Resultados guardados en:")
            print(f"   - {json_path}")
            print(f"   - {fig_path}")
        else:
            print("No se pudieron generar resultados")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()