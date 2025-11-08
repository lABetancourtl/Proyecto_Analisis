# Bibliometric Analysis Project

## Overview
This is a university project aimed at developing a bibliometric analysis and visualization tool similar to VOSviewer. The tool allows analyzing scientific publication data, extracting relationships between authors, keywords, or institutions, and visualizing them as interactive networks.

---

## Development Team
### Anderson Betancourt Arenas
### Damir Alexis Chapal I
---

## Initial Implementation Guide

Follow these instructions to set up and run the project for the first time.

### Prerequisites

  ```hash
  Python 3.8 or higher
  pip (Python package manager)
  ```

## 1. Create a Virtual Environment
  It's recommended to use a virtual environment to manage project dependencies.
  
  On Windows:
  ```bash
  python -m venv venv
  ```
  ```bash
  venv\Scripts\activate
  ```
  On Linux/macOS:
  ```bash
  python3 -m venv venv
  ```
  ```bash
  source venv/bin/activate
  ```
  
Once activated, your terminal should display (venv) before the path.

---

## 2. Install Dependencies
With the virtual environment active, install the required libraries.

### Option A: Manual Installation

 ```bash
  pip install streamlit seaborn jellyfish rapidfuzz sentence-transformers scikit-learn wordcloud plotly reportlab geopy pycountry kaleido matplotlib folium streamlit-folium
 ```
 
  
### Option B: Installation from requirements.txt (Recommended)

- We have a file with all the necessary dependencies, so you just need to run the following command and everything will be installed automatically.

  ```bash
  pip install -r requirements.txt
  ```
## Core Dependencies Description

| Library  |  Purpose |
|---|---|
|streamlit |Create interactive web apps for data science  |
|seaborn |Statistical data visualization |
|jellyfish |Approximate string matching algorithms |
|rapidfuzz |Fast string matching and similarity computation  |
|sentence |Text encoding into vector representations  |
|scikit |Machine learning and data analysis   |
|wordcloud  |Word cloud generation   |
|plotly  |Interactive charts and visualizations   |
|reportlab  |PDF report generation   |
|geopy   |Geocoding and geographic calculations   |
|pycountry  |Country data handling   |
|kaleido  |Static graph export   |
|matplotlib  |Basic data visualization   |
|folium  |Interactive map visualization and geospatial data mapping   |
|streamlit-folium  |Embed Folium interactive maps inside Streamlit apps   |
---

##  Running the Project

To run the project, open a terminal in the project root (with virtual environment activated) and execute:

```bash
streamlit run visual/indexfinal.py
```
The application will automatically open in your default browser at http://localhost:8501.

---

### Useful Commands for Linux

  ## If you encounter permission issues:
  
  ```bash
    # Give execution permissions to scripts if needed
    chmod +x scripts/*.sh
    
    # If there are system package installation issues
    sudo apt-get update
    sudo apt-get install python3-pip python3-venv
  ```
### To deactivate the virtual environment:

```bash
  deactivate
```
### To remove the virtual environment:

```bash
  rm -rf venv
```

### Project Structure

```bash
  PROYECTO_ANALISIS/
  │
  ├── visual/
  │   └── indexfinal.py      # Main application file
  ├── requirements.txt       # Project dependencies
  ├── README.md             # This file
  └── ...                   # Other project files
```
---
## Support

If you encounter any issues during installation or execution:

  1. Verify that Python is correctly installed
  2. Confirm that the virtual environment is activated
  3. Ensure you're in the project root directory
  4. Check that all dependencies installed correctly

You're all set! Now you can start using the bibliometric analysis tool.

