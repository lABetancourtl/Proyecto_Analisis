import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Para el mapa geográfico
import pycountry
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

# Para exportar a PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ScientificVisualization:
    def __init__(self, project_root):
        self.project_root = project_root
        self.df = None
        self.country_data = None
        self.word_freq = None
        self.skipped_records = 0
        
    def load_ris_data(self, ris_path):
        """
        Carga datos RIS específicamente para el Requerimiento 5
        """
        entries = []
        current_entry = {}
        
        with open(ris_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('TY  -'):
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {}
                elif line.startswith('ER  -'):
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {}
                elif '  - ' in line:
                    try:
                        tag, value = line.split('  - ', 1)
                        tag = tag.strip()
                        
                        # Mapeo de campos RIS
                        field_map = {
                            'TI': 'title',
                            'T1': 'title',
                            'AU': 'authors',
                            'A1': 'authors',
                            'PY': 'year',
                            'Y1': 'year',
                            'JO': 'journal',
                            'JF': 'journal',
                            'T2': 'journal',
                            'PB': 'publisher',
                            'CY': 'city',
                            'PP': 'publisher_city',
                            'KW': 'keywords',
                            'AB': 'abstract',
                            'N2': 'abstract',
                            'AD': 'affiliation',
                            'SN': 'issn',
                            'VL': 'volume',
                            'IS': 'issue',
                            'SP': 'pages_start',
                            'EP': 'pages_end',
                            'DO': 'doi',
                            'UR': 'url',
                            'L1': 'pdf_url',
                            'LA': 'language'
                        }
                        
                        if tag in field_map:
                            field_name = field_map[tag]
                            
                            if field_name in current_entry:
                                if isinstance(current_entry[field_name], list):
                                    current_entry[field_name].append(value.strip())
                                else:
                                    current_entry[field_name] = [current_entry[field_name], value.strip()]
                            else:
                                current_entry[field_name] = value.strip()
                                
                    except ValueError:
                        continue
        
        if current_entry:
            entries.append(current_entry)
        
        df = pd.DataFrame(entries)
        
        required_fields = ['title', 'authors', 'year', 'journal', 'abstract']
        for field in required_fields:
            if field not in df.columns:
                df[field] = ''
        
        df = self._clean_loaded_data(df)
        
        return df
    
    def _clean_loaded_data(self, df):
        """Limpia y estandariza los datos cargados"""
        if 'year' in df.columns:
            df['year'] = df['year'].apply(self._extract_year)
        
        if 'authors' in df.columns:
            df['first_author'] = df['authors'].apply(self._extract_first_author)
        
        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].fillna('')
            if 'N2' in df.columns:
                mask = (df['abstract'].isna()) | (df['abstract'] == '')
                df.loc[mask, 'abstract'] = df.loc[mask, 'N2']
        
        if 'title' in df.columns:
            df['title'] = df['title'].fillna('Sin título')
        
        return df
    
    def _extract_year(self, year_str):
        if pd.isna(year_str) or not year_str:
            return None
        
        try:
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', str(year_str))
            if year_match:
                year = int(year_match.group())
                if 1900 <= year <= datetime.now().year:
                    return year
        except:
            pass
        
        return None
    
    def _extract_first_author(self, authors_str):
        if pd.isna(authors_str) or not authors_str:
            return ''
        
        authors = str(authors_str)
        
        separators = [';', ',', ' and ', '&']
        
        for sep in separators:
            if sep in authors:
                first_author = authors.split(sep)[0].strip()
                if first_author:
                    return first_author
        
        return authors.strip()
    
    def load_data(self):
        unified_path = os.path.join(self.project_root, "resultados", "requerimiento1", "resultados_unificados.ris")
        
        if not os.path.exists(unified_path):
            raise FileNotFoundError("No se encontró el archivo unificado. Ejecuta primero el Requerimiento 1.")
        
        self.df = self.load_ris_data(unified_path)
        
        st.success(f"Cargados {len(self.df)} documentos")
        return len(self.df)
    
    def extract_geographic_data(self):
        countries = []
        valid_records = 0
        
        for idx, row in self.df.iterrows():
            if self._is_valid_record(row):
                country = self._extract_country_from_record(row)
                countries.append(country)
                valid_records += 1
            else:
                countries.append('Skipped')
                self.skipped_records += 1
        
        self.df['country'] = countries
        
        valid_country_data = self.df[self.df['country'] != 'Skipped']
        
        if len(valid_country_data) > 0:
            self.country_data = pd.DataFrame(valid_country_data['country'].value_counts()).reset_index()
            self.country_data.columns = ['country', 'count']
            
            self.country_data['country_clean'] = self.country_data['country'].apply(self._standardize_country_name)
            self.country_data = self.country_data[self.country_data['country_clean'] != 'Unknown']
        else:
            self.country_data = pd.DataFrame(columns=['country', 'count', 'country_clean'])
        
        return self.country_data
    
    def _is_valid_record(self, row):
        has_title = 'title' in row and pd.notna(row['title']) and str(row['title']).strip() != ''
        
        if not has_title:
            return False
        
        has_any_data = False
        
        if 'year' in row and pd.notna(row['year']):
            has_any_data = True
        
        if not has_any_data and 'authors' in row and pd.notna(row['authors']) and str(row['authors']).strip() != '':
            has_any_data = True
        
        if not has_any_data and 'abstract' in row and pd.notna(row['abstract']) and len(str(row['abstract']).strip()) > 10:
            has_any_data = True
        
        return has_title and has_any_data
    
    def _extract_country_from_record(self, row):
        if 'city' in row and pd.notna(row['city']) and str(row['city']).strip():
            city_country = str(row['city']).strip()
            country = self._extract_country_from_text(city_country)
            if country != 'Unknown':
                return country
        
        if 'publisher' in row and pd.notna(row['publisher']):
            publisher = str(row['publisher'])
            country = self._extract_country_from_text(publisher)
            if country != 'Unknown':
                return country
        
        if 'affiliation' in row and pd.notna(row['affiliation']):
            affiliation = str(row['affiliation'])
            country = self._extract_country_from_text(affiliation)
            if country != 'Unknown':
                return country
        
        if 'authors' in row and pd.notna(row['authors']):
            author = str(row['authors'])
            country = self._extract_country_from_author(author)
            if country != 'Unknown':
                return country
        
        if 'journal' in row and pd.notna(row['journal']):
            journal = str(row['journal'])
            country = self._extract_country_from_text(journal)
            if country != 'Unknown':
                return country
        
        return 'Unknown'
    
    def _extract_country_from_author(self, author_text):
        author_text = author_text.lower()
        
        patterns = {
            'university of california': 'United States',
            'stanford university': 'United States',
            'mit': 'United States',
            'harvard': 'United States',
            'oxford': 'United Kingdom',
            'cambridge': 'United Kingdom',
            'university of toronto': 'Canada',
            'university of sydney': 'Australia',
            'university of tokyo': 'Japan',
            'tsinghua': 'China',
            'peking university': 'China',
            'eth zurich': 'Switzerland',
            'max planck': 'Germany',
            'technical university': 'Germany',
            'university of sao paulo': 'Brazil',
            'iit': 'India',
            'national university of singapore': 'Singapore'
        }
        
        for pattern, country in patterns.items():
            if pattern in author_text:
                return country
        
        return 'Unknown'
    
    def _extract_country_from_text(self, text):
        text = str(text).lower()
        
        country_keywords = {
            'usa': 'United States', 'united states': 'United States', 'us': 'United States', 'u.s.': 'United States',
            'china': 'China', 'peoples republic of china': 'China', 'chinese': 'China',
            'uk': 'United Kingdom', 'united kingdom': 'United Kingdom', 'england': 'United Kingdom', 'british': 'United Kingdom',
            'germany': 'Germany', 'deutschland': 'Germany', 'german': 'Germany',
            'france': 'France', 'french': 'France',
            'canada': 'Canada', 'canadian': 'Canada',
            'australia': 'Australia', 'australian': 'Australia',
            'japan': 'Japan', 'japanese': 'Japan',
            'spain': 'Spain', 'spanish': 'Spain',
            'italy': 'Italy', 'italian': 'Italy',
            'brazil': 'Brazil', 'brasil': 'Brazil', 'brazilian': 'Brazil',
            'india': 'India', 'indian': 'India',
            'korea': 'South Korea', 'south korea': 'South Korea', 'korean': 'South Korea',
            'netherlands': 'Netherlands', 'holland': 'Netherlands', 'dutch': 'Netherlands',
            'switzerland': 'Switzerland', 'swiss': 'Switzerland',
            'sweden': 'Sweden', 'swedish': 'Sweden',
            'mexico': 'Mexico', 'mexican': 'Mexico',
            'argentina': 'Argentina', 'argentinian': 'Argentina',
            'chile': 'Chile', 'chilean': 'Chile',
            'colombia': 'Colombia', 'colombian': 'Colombia',
            'russia': 'Russia', 'russian': 'Russia'
        }
        
        for keyword, country in country_keywords.items():
            if keyword in text:
                return country
        
        return 'Unknown'
    
    def _standardize_country_name(self, country_name):
        if country_name == 'Unknown':
            return 'Unknown'
        
        try:
            country = pycountry.countries.search_fuzzy(country_name)
            return country[0].name
        except:
            return country_name
    
    def create_heatmap(self):
        if self.country_data is None:
            self.extract_geographic_data()
        
        if len(self.country_data) == 0:
            st.warning("No hay datos geográficos suficientes para generar el mapa de calor")
            return None
        
        country_counts = self.country_data.groupby('country_clean')['count'].sum().reset_index()
        
        fig = px.choropleth(
            country_counts,
            locations='country_clean',
            locationmode='country names',
            color='count',
            hover_name='country_clean',
            hover_data={'count': True, 'country_clean': False},
            color_continuous_scale='Viridis',
            title='Distribución Geográfica de Publicaciones por País del Primer Autor'
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=500
        )
        
        return fig
    
    def generate_wordcloud_data(self):
        all_text = []
        
        for idx, row in self.df.iterrows():
            if 'abstract' in row and pd.notna(row['abstract']) and str(row['abstract']).strip():
                abstract = str(row['abstract']).strip()
                if len(abstract) > 10:
                    all_text.append(abstract)
            
            if 'keywords' in row and pd.notna(row['keywords']) and str(row['keywords']).strip():
                keywords = str(row['keywords']).strip()
                all_text.append(keywords)
            
            if 'title' in row and pd.notna(row['title']) and str(row['title']).strip():
                title = str(row['title']).strip()
                if len(title) > 5:
                    all_text.append(title)
        
        if len(all_text) == 0:
            st.warning("No hay textos válidos para generar la nube de palabras")
            return None
        
        full_text = ' '.join(all_text)
        
        import re
        from collections import Counter
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())
        
        stopwords = set([
            'this', 'that', 'with', 'from', 'have', 'has', 'had', 'were', 'which', 
            'their', 'what', 'will', 'would', 'about', 'when', 'them', 'some', 
            'into', 'such', 'than', 'then', 'also', 'more', 'most', 'these', 
            'there', 'other', 'using', 'based', 'study', 'research', 'paper',
            'results', 'method', 'approach', 'model', 'system', 'data', 'analysis',
            'however', 'therefore', 'addition', 'further', 'different', 'various'
        ])
        
        filtered_words = [word for word in words if word not in stopwords]
        
        if len(filtered_words) == 0:
            return None
        
        self.word_freq = Counter(filtered_words)
        return self.word_freq
    
    def create_wordcloud(self, max_words=100):
        if self.word_freq is None:
            word_freq = self.generate_wordcloud_data()
            if word_freq is None:
                st.warning("No se pudo generar la nube de palabras por falta de datos válidos")
                return None
        
        wordcloud_dict = dict(self.word_freq.most_common(max_words))
        
        if len(wordcloud_dict) == 0:
            st.warning("No hay palabras suficientes para generar la nube")
            return None
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=max_words
        ).generate_from_frequencies(wordcloud_dict)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Nube de Palabras - Términos más Frecuentes en Abstracts y Keywords', 
                    fontsize=14, pad=20)
        
        return fig
    
    def create_timeline(self):
        years = []
        journals = []
        
        for idx, row in self.df.iterrows():
            year = None
            if 'year' in row and pd.notna(row['year']):
                year = row['year']
            
            journal = 'Unknown'
            if 'journal' in row and pd.notna(row['journal']) and str(row['journal']).strip():
                journal = str(row['journal']).strip()
            
            if year is not None:
                years.append(year)
                journals.append(journal)
        
        if len(years) == 0:
            st.warning("No hay registros con datos de año para la línea temporal")
            return None
        
        timeline_data = pd.DataFrame({
            'year': years,
            'journal': journals
        })
        
        timeline_summary = timeline_data.groupby(['year', 'journal']).size().reset_index()
        timeline_summary.columns = ['year', 'journal', 'count']
        
        fig = px.line(
            timeline_summary,
            x='year',
            y='count',
            color='journal',
            title=f'Línea Temporal de Publicaciones por Año y Revista ({len(years)} registros con año)',
            labels={'year': 'Año', 'count': 'Número de Publicaciones', 'journal': 'Revista'}
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            xaxis=dict(tickmode='linear')
        )
        
        return fig
    
    def export_to_pdf(self, output_dir):
        """Exporta los tres gráficos a PDF - VERSIÓN MEJORADA"""
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, "analisis_visual.pdf")
        
        try:
            # Crear documento PDF
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1
            )
            
            story.append(Paragraph("Análisis Visual de Producción Científica", title_style))
            story.append(Spacer(1, 20))
            
            # Guardar gráficos temporalmente
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 1. Mapa de calor
            story.append(Paragraph("1. Mapa de Calor Geográfico", styles['Heading2']))
            map_fig = self.create_heatmap()
            if map_fig:
                map_path = os.path.join(temp_dir, "mapa_calor.png")
                map_fig.write_image(map_path, width=800, height=400)
                story.append(Image(map_path, width=6*inch, height=3*inch))
            else:
                story.append(Paragraph("No hay datos suficientes para el mapa de calor", styles['Italic']))
            story.append(Spacer(1, 20))
            
            # 2. Nube de palabras
            story.append(Paragraph("2. Nube de Palabras", styles['Heading2']))
            wc_fig = self.create_wordcloud()
            if wc_fig:
                wc_path = os.path.join(temp_dir, "nube_palabras.png")
                wc_fig.savefig(wc_path, bbox_inches='tight', dpi=150, facecolor='white')
                story.append(Image(wc_path, width=6*inch, height=3*inch))
            else:
                story.append(Paragraph("No hay datos suficientes para la nube de palabras", styles['Italic']))
            story.append(Spacer(1, 20))
            
            # 3. Línea temporal
            story.append(Paragraph("3. Línea Temporal", styles['Heading2']))
            timeline_fig = self.create_timeline()
            if timeline_fig:
                timeline_path = os.path.join(temp_dir, "linea_temporal.png")
                timeline_fig.write_image(timeline_path, width=800, height=400)
                story.append(Image(timeline_path, width=6*inch, height=3*inch))
            else:
                story.append(Paragraph("No hay datos suficientes para la línea temporal", styles['Italic']))
            
            # Construir PDF
            doc.build(story)
            
            # Limpiar archivos temporales
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return pdf_path
            
        except Exception as e:
            st.error(f"Error al generar PDF: {str(e)}")
            return None

def mostrar_requerimiento_5(project_root):
    st.markdown('<div class="requirement-title">Requerimiento 5: Visualizaciones Avanzadas</div>', unsafe_allow_html=True)
    
    st.write("""
    Este requerimiento genera visualizaciones avanzadas de la producción científica:
    - **Mapa de calor geográfico** con distribución por países
    - **Nube de palabras** dinámica de términos más frecuentes  
    - **Línea temporal** de publicaciones por año y revista
    - **Exportación a PDF** de todos los análisis
    """)
    
    # Inicializar session_state si no existe
    if 'visualizations_generated' not in st.session_state:
        st.session_state.visualizations_generated = False
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # Inicializar analizador
    if st.session_state.analyzer is None:
        st.session_state.analyzer = ScientificVisualization(project_root)
    
    analyzer = st.session_state.analyzer
    
    try:
        # Cargar datos (siempre se carga, no depende del botón)
        n_docs = analyzer.load_data()
        
        # Configuración
        st.subheader("Configuración de Visualizaciones")
        
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider("Máximo de palabras en nube", 50, 200, 100)
        with col2:
            show_raw_data = st.checkbox("Mostrar datos brutos", value=False)
        
        # Botón principal para generar visualizaciones
        if st.button("Generar Visualizaciones", type="primary", key="generate_viz"):
            with st.spinner("Generando análisis visual..."):
                st.session_state.visualizations_generated = True
                
                # 1. MAPA DE CALOR
                st.subheader("Mapa de Calor Geográfico")
                st.write("Distribución de publicaciones por país del primer autor")
                
                analyzer.extract_geographic_data()
                map_fig = analyzer.create_heatmap()
                
                if map_fig:
                    st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.warning("No hay datos geográficos suficientes para generar el mapa")
                
                if show_raw_data and analyzer.country_data is not None:
                    st.write("**Datos geográficos:**")
                    st.dataframe(analyzer.country_data)
                
                # 2. NUBE DE PALABRAS
                st.subheader("Nube de Palabras")
                st.write("Términos más frecuentes en abstracts y keywords")
                
                wc_fig = analyzer.create_wordcloud(max_words=max_words)
                
                if wc_fig:
                    st.pyplot(wc_fig)
                    
                    # Mostrar tabla de frecuencias
                    word_freq = analyzer.generate_wordcloud_data()
                    if word_freq:
                        top_words = pd.DataFrame(
                            word_freq.most_common(20), 
                            columns=['Palabra', 'Frecuencia']
                        )
                        st.write("**Top 20 palabras más frecuentes:**")
                        st.dataframe(top_words)
                else:
                    st.warning("No hay textos suficientes para generar la nube de palabras")
                
                # 3. LÍNEA TEMPORAL
                st.subheader("Línea Temporal de Publicaciones")
                st.write("Evolución de publicaciones por año y revista")
                
                timeline_fig = analyzer.create_timeline()
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos de años para generar la línea temporal")
        
        # Mostrar sección de exportación a PDF SOLO si las visualizaciones fueron generadas
        if st.session_state.visualizations_generated:
            st.subheader("Exportar a PDF")
            
            # Botón separado para PDF
            if st.button("Generar Reporte PDF", key="generate_pdf"):
                with st.spinner("Generando PDF..."):
                    output_dir = os.path.join(project_root, "resultados", "requerimiento5")
                    pdf_path = analyzer.export_to_pdf(output_dir)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        st.success(f"✅ PDF generado exitosamente: {pdf_path}")
                        
                        # Mostrar enlace de descarga
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        
                        st.download_button(
                            label="Descargar PDF",
                            data=pdf_bytes,
                            file_name="analisis_visual_produccion_cientifica.pdf",
                            mime="application/pdf",
                            key="download_pdf"
                        )
                    else:
                        st.error("No se pudo generar el PDF")
    
    except FileNotFoundError as e:
        st.error(f"{str(e)}")
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")

if __name__ == "__main__":
    main()