import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from requerimiento1.req1_unificacion import mostrar_requerimiento_1
from requerimiento2.req2_similitud import mostrar_requerimiento_2
from requerimiento3.req3_palabras import mostrar_requerimiento_3
from requerimiento4.requerimiento4_clustering import mostrar_requerimiento_4


# === CONFIGURACIÓN GENERAL ===

def get_project_root():
    """Devuelve la ruta raíz del proyecto"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = get_project_root()

st.set_page_config(
    page_title="Análisis Bibliométrico - Generative Artificial Intelligence",
    page_icon="📊",
    layout="wide"
)

# === ESTILOS CSS GLOBALES ===
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 25px;
    }
    .requirement-title {
        font-size: 1.4em;
        font-weight: bold;
        color: #3498db;
        margin: 20px 0 15px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #3498db;
    }
    .description {
        color: #555;
        font-size: 1.1em;
        text-align: justify;
    }
    </style>
""", unsafe_allow_html=True)

# === FUNCIÓN PRINCIPAL ===
def main():
    st.markdown('<div class="main-title">📘 Análisis Bibliométrico<br>Inteligencia Artificial Generativa</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="description">
    Este sistema realiza un análisis bibliométrico sobre publicaciones relacionadas con 
    <strong>"Generative Artificial Intelligence"</strong>, utilizando múltiples fuentes científicas 
    y aplicando técnicas de análisis de similitud textual clásicas y con modelos de IA.
    </div>
    """, unsafe_allow_html=True)

    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "Requerimiento 1: Unificación y Duplicados",
        "Requerimiento 2: Similitud Textual",
        "Requerimiento 3: Palabras Clave y Frecuencia",
        "Requerimiento 4: Dendrograma de Similitud"
    ])


    # === 📂 PESTAÑA 1 ===
    with tab1:
        req1_results_dir = os.path.join(PROJECT_ROOT, "resultados", "requerimiento1")
        req1_script = os.path.join(PROJECT_ROOT, "requerimiento1", "scrapy", "MainScrapys.py")
        mostrar_requerimiento_1(req1_results_dir, req1_script)

    # === ⚙️ PESTAÑA 2 ===
    with tab2:
        mostrar_requerimiento_2(PROJECT_ROOT)

    # === 🗝️ PESTAÑA 3 ===
    with tab3:
        mostrar_requerimiento_3(PROJECT_ROOT)

    with tab4:
        mostrar_requerimiento_4(PROJECT_ROOT)

# === PUNTO DE ENTRADA ===
if __name__ == "__main__":
    main()
