# Bibliometric Analysis Project

## Overview
This is a university project aimed at developing a bibliometric analysis and visualization tool similar to **VOSviewer**.  
The goal is to analyze scientific publication data, extract relationships between authors, keywords, or institutions, and visualize them as interactive networks.

---

## Team
### Anderson Betancourt Arenas
### Damir Alexis Chapal I
---
## The following instructions will assist in the correct initial implementation to run the project for the first time.
### Create a Virtual Environment
  In your project directory:
  ```bash
  python -m venv venv
  ```
  Then:
  ```bash
  venv\Scripts\activate
  ```
Once activated, your terminal should display (venv) before the path.

---

### Install Required Libraries
With the virtual environment active, install the core dependencies and libraries.

There are two ways to do it:
## Install everything manually

  ```bash
  pip install streamlit
  ```
  ```bash
 pip install seaborn
  ```
  ```bash
  pip install jellyfish
  ```
  ```bash
  pip install rapidfuzz
  ```
  ```bash
  pip install sentence-transformers
  ```
  ```bash
  pip install scikit-learn
  ```
  ```bash
  pip install wordcloud
  ```
  ```bash
  pip install plotly
  ```
  ```bash
  pip install reportlab
  ```
  ```bash
  pip install geopy
  ```
  ```bash
  pip install pycountry
  ```
  ```bash
  pip install kaleido
  ```
  ```bash
  pip install matplotlib
  ```
  
## Install with requeriments.txt
- We have a file with all the necessary dependencies, so you just need to run the following command and everything will be installed automatically.

  ```bash
  pip install -r requirements.txt
  ```
## Dependencies Description

- wordcloud → Used to generate visual word clouds from text data.
- seaborn → A statistical data visualization library built on top of Matplotlib.
- jellyfish → Provides algorithms for approximate and phonetic string matching.
- rapidfuzz → A fast string matching and similarity computation library.
- sentence-transformers → Allows encoding of sentences and paragraphs into dense vector representations for semantic similarity and clustering.
- scikit-learn → A powerful machine learning library for data preprocessing, classification, regression, and clustering.
- streamlit → Used to build interactive web apps for data science and machine learning projects easily.

Once the dependecies were installed we can now run the project

---

## How to run the project

To run the project, we just need to execute one instruction in the root of our project, so the first step is to open a new terminal in our project root and then we need to execute this:

```bash
streamlit run visual/indexfinal.py
```
If you have followed all the steps correctly since installing the virtual environment, then the project should now be running.
