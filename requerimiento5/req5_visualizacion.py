import streamlit as st
import os
import sys

# Agregar ruta para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from requerimiento5.requerimiento5_visualizacion import mostrar_requerimiento_5

def main():
    st.set_page_config(
        page_title="Requerimiento 5 - Visualización",
        page_icon="📊",
        layout="wide"
    )
    
    # Estilos CSS
    st.markdown("""
        <style>
        .requirement-title {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
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
    
    mostrar_requerimiento_5(project_root)

if __name__ == "__main__":
    main()