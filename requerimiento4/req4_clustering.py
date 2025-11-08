import streamlit as st
import os
import sys

# Agregar ruta para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from requerimiento4.requerimiento4_clustering import mostrar_requerimiento_4

def main():
    st.set_page_config(
        page_title="Requerimiento 4 - Clustering",
        page_icon="🌳",
        layout="wide"
    )
    
    # Estilos CSS
    st.markdown("""
        <style>
        .requirement-title {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    mostrar_requerimiento_4(project_root)

if __name__ == "__main__":
    main()