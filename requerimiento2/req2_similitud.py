import streamlit as st
import os
import pandas as pd
from requerimiento2.requerimiento2_similitud import load_unified_ris, compute_similarities, save_sim_results

def mostrar_requerimiento_2(project_root):
    st.markdown('<div class="requirement-title">Requerimiento 2: Similitud Textual</div>', unsafe_allow_html=True)
    st.write("Selecciona dos o más artículos del archivo unificado para calcular la similitud textual usando distintos algoritmos.")

    unified_path = os.path.join(project_root, "resultados", "requerimiento1", "resultados_unificados.ris")

    if not os.path.exists(unified_path):
        st.warning("⚠️ No se encontró el archivo unificado de Requerimiento 1. Ejecútalo primero.")
        return

    df = load_unified_ris(unified_path)
    titles = df['TI'].fillna('Sin título').tolist()

    st.subheader("Selecciona artículos para comparar:")
    selected_indices = st.multiselect(
        "Selecciona 2 o más artículos:",
        options=list(range(len(titles))),
        format_func=lambda i: f"{i+1}. {titles[i][:120]}"
    )

    if len(selected_indices) < 2:
        st.info("Selecciona al menos dos artículos para continuar.")
        return

    texts = [df.iloc[i]['AB'] if pd.notna(df.iloc[i]['AB']) else df.iloc[i]['TI'] for i in selected_indices]

    # --- multiselect con nombres amigables ---
    friendly_methods = st.multiselect(
        "Selecciona los algoritmos de similitud:",
        ['Levenshtein', 'Jaro-Winkler', 'Jaccard', 'TF-IDF', 'SBERT'],
        default=['Levenshtein', 'TF-IDF', 'SBERT']
    )

    # --- mapear a claves internas ---
    method_map = {
        'Levenshtein': 'lev',
        'Jaro-Winkler': 'jw',
        'Jaccard': 'jaccard',
        'TF-IDF': 'tfidf',
        'SBERT': 'sbert'
    }
    selected_methods = [method_map[m] for m in friendly_methods]

    if st.button("🔍 Calcular similitud"):
        with st.spinner("Calculando similitudes..."):
            # 1️⃣ CALCULO DE SIMILITUD
            results = compute_similarities(texts, selected_methods)
            
            # 2️⃣ VISUALIZACIÓN DE RESULTADOS (AGREGAR AQUÍ)
            # ================================================
            import seaborn as sns
            import matplotlib.pyplot as plt

            # --- Tabla resumen ---
            pairs = results['pairs']
            tfidf_matrix = results['matrices'].get('tfidf', [])
            sbert_matrix = results['matrices'].get('sbert', [])

            data = []
            for pair in pairs:
                i, j = pair['i'], pair['j']
                detail = results['details'].get(f"{i}-{j}", {})
                levenshtein_score = detail.get('levenshtein', {}).get('score', None)
                data.append({
                    'Artículos': pair['title'],
                    'TF-IDF': round(tfidf_matrix[i][j], 3) if tfidf_matrix else None,
                    'SBERT': round(sbert_matrix[i][j], 3) if sbert_matrix else None,
                    'Levenshtein': round(levenshtein_score, 3) if levenshtein_score is not None else None
                })
            df_results = pd.DataFrame(data)
            st.subheader("📊 Tabla de Similitudes")
            st.table(df_results)

            # --- Heatmaps ---
            if tfidf_matrix:
                st.subheader("Heatmap TF-IDF")
                fig, ax = plt.subplots()
                sns.heatmap(tfidf_matrix, annot=True, cmap='coolwarm', xticklabels=[f"A{i}" for i in range(len(tfidf_matrix))],
                            yticklabels=[f"A{i}" for i in range(len(tfidf_matrix))])
                st.pyplot(fig)

            if sbert_matrix:
                st.subheader("Heatmap SBERT")
                fig, ax = plt.subplots()
                sns.heatmap(sbert_matrix, annot=True, cmap='viridis', xticklabels=[f"A{i}" for i in range(len(sbert_matrix))],
                            yticklabels=[f"A{i}" for i in range(len(sbert_matrix))])
                st.pyplot(fig)

            # --- Detalles paso a paso ---
            st.subheader("Detalles de los Algoritmos")
            st.write("TF-IDF:", results.get('tfidf_steps', {}))
            st.write("SBERT:", results.get('sbert_steps', {}))
            for pair in pairs:
                i, j = pair['i'], pair['j']
                lev_detail = results['details'].get(f"{i}-{j}", {}).get('levenshtein', {})
                st.write(f"Levenshtein {pair['title']}:", lev_detail)
            # ================================================

            # 3️⃣ GUARDAR RESULTADOS
            out_dir = os.path.join(project_root, "resultados", "requerimiento2")
            saved = save_sim_results(out_dir, results)
            st.success(f"✅ Resultados guardados en {saved}")

