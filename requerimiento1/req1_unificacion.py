import streamlit as st
import os
import json
from PIL import Image
import subprocess

# === Funciones auxiliares del requerimiento 1 ===

def safe_json_load(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error cargando {filepath}: {str(e)}")
        return {}

def safe_image_load(filepath):
    try:
        return Image.open(filepath)
    except Exception as e:
        st.error(f"Error cargando imagen {filepath}: {str(e)}")
        return None

def run_script(path, label=None):
    if label:
        st.info(f"Ejecutando: {label}")
    try:
        subprocess.run(["python", path], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error al ejecutar {label or path}: {e}")
        raise e


# === Vista principal del requerimiento 1 ===

def mostrar_requerimiento_1(results_dir, script_path):
    """Renderiza en Streamlit la sección del Requerimiento 1"""
    st.markdown('<div class="requirement-title">Requerimiento 1: Unificación de Datos y Eliminación de Duplicados</div>', unsafe_allow_html=True)

    if st.button('Ejecutar Requerimiento 1'):
        with st.spinner('Ejecutando Scrapy y unificación...'):
            try:
                run_script(script_path, "MainScrapys.py")
                st.success("✅ Requerimiento 1 ejecutado con éxito.")
            except Exception as e:
                st.error(f"Error en Requerimiento 1: {e}")

    # Mostrar resultados si existen
    stats_path = os.path.join(results_dir, "estadisticas_req1.json")
    unificado_path = os.path.join(results_dir, "resultados_unificados.ris")
    duplicados_path = os.path.join(results_dir, "duplicados_eliminados.ris")

    if not os.path.exists(unificado_path):
        st.warning("⚠️ Aún no existe un archivo unificado. Ejecuta el requerimiento 1 primero.")
        return

    stats = safe_json_load(stats_path)
    st.markdown("### 📂 Resultados de la unificación")

    st.write(f"- Total de registros: {stats.get('total_records', 'N/A')}")
    st.write(f"- Duplicados eliminados: {stats.get('duplicates', 'N/A')}")
    st.write(f"- Registros únicos: {stats.get('unique_records', 'N/A')}")

    col1, col2 = st.columns(2)
    with col1:
        with open(unificado_path, 'rb') as f:
            st.download_button("⬇Descargar Archivo Unificado", f, file_name="resultados_unificados.ris", mime="application/ris")
    with col2:
        if os.path.exists(duplicados_path):
            with open(duplicados_path, 'rb') as f:
                st.download_button("⬇Descargar Duplicados Eliminados", f, file_name="duplicados_eliminados.ris", mime="application/ris")
