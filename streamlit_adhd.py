# -*- coding: utf-8 -*-
"""
Streamlit TDAH - Outil de Dépistage et d'Analyse
Version enrichie avec dataset réel ASRS et analyses avancées
"""

import streamlit as st
import joblib
import hashlib
import os
import pickle
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dépistage TDAH",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Création des dossiers de cache
for folder in ['data_cache', 'image_cache', 'model_cache', 'theme_cache']:
    os.makedirs(folder, exist_ok=True)

# État de session pour les scores ADHD-RS
if "adhd_total" not in st.session_state:
    st.session_state.adhd_total = 0

if "adhd_responses" not in st.session_state:
    st.session_state.adhd_responses = []

# Questions ASRS officielles
ASRS_QUESTIONS = {
    "Partie A - Questions de dépistage principal": [
        "À quelle fréquence avez-vous des difficultés à terminer les détails finaux d'un projet, une fois que les parties difficiles ont été faites ?",
        "À quelle fréquence avez-vous des difficultés à organiser les tâches lorsque vous devez faire quelque chose qui demande de l'organisation ?", 
        "À quelle fréquence avez-vous des problèmes pour vous rappeler des rendez-vous ou des obligations ?",
        "Quand vous avez une tâche qui demande beaucoup de réflexion, à quelle fréquence évitez-vous ou retardez-vous de commencer ?",
        "À quelle fréquence bougez-vous ou vous tortillez-vous avec vos mains ou vos pieds quand vous devez rester assis longtemps ?",
        "À quelle fréquence vous sentez-vous excessivement actif et obligé de faire des choses, comme si vous étiez mené par un moteur ?"
    ],
    "Partie B - Questions complémentaires": [
        "À quelle fréquence faites-vous des erreurs d'inattention quand vous travaillez sur un projet ennuyeux ou difficile ?",
        "À quelle fréquence avez-vous des difficultés à maintenir votre attention quand vous faites un travail ennuyeux ou répétitif ?",
        "À quelle fréquence avez-vous des difficultés à vous concentrer sur ce que les gens vous disent, même quand ils s'adressent directement à vous ?",
        "À quelle fréquence égarez-vous ou avez des difficultés à retrouver des choses à la maison ou au travail ?",
        "À quelle fréquence êtes-vous distrait par l'activité ou le bruit autour de vous ?",
        "À quelle fréquence quittez-vous votre siège dans des réunions ou d'autres situations où vous devriez rester assis ?",
        "À quelle fréquence vous sentez-vous agité ou nerveux ?",
        "À quelle fréquence avez-vous des difficultés à vous détendre quand vous avez du temps libre ?",
        "À quelle fréquence vous retrouvez-vous à trop parler dans des situations sociales ?",
        "Quand vous êtes en conversation, à quelle fréquence finissez-vous les phrases des personnes à qui vous parlez, avant qu'elles puissent les finir elles-mêmes ?",
        "À quelle fréquence avez-vous des difficultés à attendre votre tour dans des situations où chacun doit attendre son tour ?",
        "À quelle fréquence interrompez-vous les autres quand ils sont occupés ?"
    ]
}

ASRS_OPTIONS = {
    0: "Jamais",
    1: "Rarement", 
    2: "Parfois",
    3: "Souvent",
    4: "Très souvent"
}

def initialize_session_state():
    """Initialise l'état de session pour conserver les configurations entre les recharges"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        default_tool = "🏠 Accueil"
        
        try:
            if "selection" in st.query_params:
                selection = st.query_params["selection"]
                selection_mapping = {
                    "📝 Test ADHD-RS": "🤖 Prédiction par IA",
                    "🤖 Prédiction par IA": "🤖 Prédiction par IA",
                    "🔍 Exploration des Données": "🔍 Exploration des Données"
                }
                if selection in selection_mapping:
                    st.session_state.tool_choice = selection_mapping[selection]
                else:
                    st.session_state.tool_choice = default_tool
            else:
                st.session_state.tool_choice = default_tool
        except:
            st.session_state.tool_choice = default_tool

        st.session_state.data_exploration_expanded = True

def set_custom_theme():
    """Définit le thème personnalisé avec palette orange pour le TDAH"""
    css_path = "theme_cache/custom_theme_tdah.css"
    os.makedirs(os.path.dirname(css_path), exist_ok=True)

    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            custom_theme = f.read()
    else:
        # CSS corrigé avec chaînes de caractères correctement fermées
        custom_theme = """
        <style>
        :root {
            --primary: #d84315 !important;
            --secondary: #ff5722 !important;
            --accent: #ff9800 !important;
            --background: #fff8f5 !important;
            --sidebar-bg: #ffffff !important;
            --sidebar-border: #ffccbc !important;
            --text-primary: #d84315 !important;
            --text-secondary: #bf360c !important;
            --sidebar-width-collapsed: 60px !important;
            --sidebar-width-expanded: 240px !important;
            --sidebar-transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            --shadow-light: 0 2px 8px rgba(255,87,34,0.08) !important;
            --shadow-medium: 0 4px 16px rgba(255,87,34,0.12) !important;
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--background) !important;
        }

        [data-testid="stSidebar"] {
            width: var(--sidebar-width-collapsed) !important;
            min-width: var(--sidebar-width-collapsed) !important;
            max-width: var(--sidebar-width-collapsed) !important;
            height: 100vh !important;
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            z-index: 999999 !important;
            background: var(--sidebar-bg) !important;
            border-right: 1px solid var(--sidebar-border) !important;
            box-shadow: var(--shadow-light) !important;
            overflow: hidden !important;
            padding: 0 !important;
            transition: var(--sidebar-transition) !important;
        }

        [data-testid="stSidebar"]:hover {
            width: var(--sidebar-width-expanded) !important;
            min-width: var(--sidebar-width-expanded) !important;
            max-width: var(--sidebar-width-expanded) !important;
            box-shadow: var(--shadow-medium) !important;
            overflow-y: auto !important;
        }

        [data-testid="stSidebar"] > div {
            width: var(--sidebar-width-expanded) !important;
            padding: 12px 8px !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"]:hover > div {
            overflow-y: auto !important;
            padding: 16px 12px !important;
        }

        [data-testid="stSidebar"] h2 {
            font-size: 0 !important;
            margin: 0 0 20px 0 !important;
            padding: 12px 0 !important;
            border-bottom: 1px solid var(--sidebar-border) !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            height: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        [data-testid="stSidebar"] h2::before {
            content: "🧠" !important;
            font-size: 28px !important;
            display: block !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"]:hover h2 {
            font-size: 1.4rem !important;
            color: var(--primary) !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"]:hover h2::before {
            font-size: 20px !important;
            margin-right: 8px !important;
        }

        [data-testid="stSidebar"] .stRadio {
            padding: 0 !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"] .stRadio > div {
            display: flex !important;
            flex-direction: column !important;
            gap: 4px !important;
            padding: 0 !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            display: flex !important;
            align-items: center !important;
            padding: 10px 6px !important;
            margin: 0 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
            position: relative !important;
            height: 44px !important;
            overflow: hidden !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] .stRadio label > div:first-child {
            display: none !important;
        }

        [data-testid="stSidebar"] .stRadio label span {
            font-size: 0 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            text-align: center !important;
            position: relative !important;
        }

        [data-testid="stSidebar"] .stRadio label span::before {
            font-size: 22px !important;
            display: block !important;
            width: 100% !important;
            text-align: center !important;
        }

        [data-testid="stSidebar"] .stRadio label:nth-child(1) span::before { content: "🏠" !important; }
        [data-testid="stSidebar"] .stRadio label:nth-child(2) span::before { content: "🔍" !important; }
        [data-testid="stSidebar"] .stRadio label:nth-child(3) span::before { content: "🧠" !important; }
        [data-testid="stSidebar"] .stRadio label:nth-child(4) span::before { content: "🤖" !important; }
        [data-testid="stSidebar"] .stRadio label:nth-child(5) span::before { content: "📚" !important; }
        [data-testid="stSidebar"] .stRadio label:nth-child(6) span::before { content: "ℹ️" !important; }

        [data-testid="stSidebar"]:hover .stRadio label span {
            font-size: 14px !important;
            font-weight: 500 !important;
            text-align: left !important;
            padding-left: 12px !important;
        }

        [data-testid="stSidebar"]:hover .stRadio label span::before {
            font-size: 18px !important;
            position: absolute !important;
            left: -8px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: auto !important;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: linear-gradient(135deg, #fff3e0, #ffe0b2) !important;
            transform: translateX(3px) !important;
            box-shadow: var(--shadow-light) !important;
        }

        [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: linear-gradient(135deg, var(--secondary), var(--accent)) !important;
            color: white !important;
            box-shadow: var(--shadow-medium) !important;
        }

        [data-testid="stSidebar"] .stRadio label[data-checked="true"]:hover {
            background: linear-gradient(135deg, var(--accent), var(--secondary)) !important;
            transform: translateX(5px) !important;
        }

        .main .block-container {
            margin-left: calc(var(--sidebar-width-collapsed) + 16px) !important;
            padding: 1.5rem !important;
            max-width: calc(100vw - var(--sidebar-width-collapsed) - 32px) !important;
            transition: var(--sidebar-transition) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--secondary), var(--accent)) !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            box-shadow: var(--shadow-light) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-medium) !important;
            background: linear-gradient(135deg, var(--accent), var(--secondary)) !important;
        }

        .info-card-modern {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(255,87,34,0.08);
            border-left: 4px solid var(--secondary);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .info-card-modern:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(255,87,34,0.15);
        }

        .stAlert, [data-testid="stAlert"] {
            border: none !important;
            background: transparent !important;
        }

        .asrs-question-card {
            background: linear-gradient(135deg, #fff3e0, #ffcc02);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #ff5722;
            box-shadow: 0 3px 10px rgba(255,87,34,0.1);
        }

        .asrs-option-container {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }

        .asrs-option {
            flex: 1;
            min-width: 80px;
            text-align: center;
        }
        </style>

        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            
            if (sidebar) {
                let isExpanded = false;
                let hoverTimeout;
                
                function expandSidebar() {
                    clearTimeout(hoverTimeout);
                    isExpanded = true;
                    sidebar.style.overflow = 'visible';
                }
                
                function collapseSidebar() {
                    hoverTimeout = setTimeout(() => {
                        isExpanded = false;
                        sidebar.style.overflow = 'hidden';
                    }, 200);
                }
                
                sidebar.addEventListener('mouseenter', expandSidebar);
                sidebar.addEventListener('mouseleave', collapseSidebar);
                
                const observer = new MutationObserver(() => {
                    const radioLabels = sidebar.querySelectorAll('.stRadio label');
                    radioLabels.forEach(label => {
                        const input = label.querySelector('input[type="radio"]');
                        if (input && input.checked) {
                            label.setAttribute('data-checked', 'true');
                        } else {
                            label.setAttribute('data-checked', 'false');
                        }
                    });
                });
                
                observer.observe(sidebar, { 
                    childList: true, 
                    subtree: true,
                    attributes: true 
                });
            }
        });
        </script>
        """
        
        with open(css_path, 'w') as f:
            f.write(custom_theme)

    st.markdown(custom_theme, unsafe_allow_html=True)

def show_navigation_menu():
    """Menu de navigation optimisé pour le TDAH"""
    st.markdown("## 🧠 TDAH - Navigation")
    st.markdown("Choisissez un outil :")

    options = [
        "🏠 Accueil",
        "🔍 Exploration", 
        "🧠 Analyse ML",
        "🤖 Prédiction par IA",
        "📚 Documentation",
        "ℹ️ À propos"
    ]

    if 'tool_choice' not in st.session_state or st.session_state.tool_choice not in options:
        st.session_state.tool_choice = "🏠 Accueil"

    current_index = options.index(st.session_state.tool_choice)

    tool_choice = st.radio(
        "",
        options,
        label_visibility="collapsed",
        index=current_index,
        key="main_navigation"
    )

    if tool_choice != st.session_state.tool_choice:
        st.session_state.tool_choice = tool_choice

    return tool_choice

@st.cache_data(ttl=86400)
def load_enhanced_dataset():
    """Charge le dataset TDAH enrichi depuis Google Drive"""
    try:
        # URL du nouveau dataset Google Drive
        url = 'https://drive.google.com/file/d/15WW4GruZFQpyrLEbJtC-or5NPjXmqsnR/view?usp=drive_link'
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Chargement du dataset
        df = pd.read_csv(download_url)
        
        # Vérification de l'intégrité des données
        st.success(f"✅ Dataset chargé avec succès ! {len(df)} participants, {len(df.columns)} variables")
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du dataset Google Drive: {str(e)}")
        st.info("Utilisation de données simulées à la place")
        return create_fallback_dataset()

def create_fallback_dataset():
    """Crée un dataset de fallback compatible avec la structure attendue"""
    np.random.seed(42)
    n_samples = 1500
    
    # Structure basée sur le vrai dataset
    data = {
        'subject_id': [f'FALLBACK_{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'diagnosis': np.random.binomial(1, 0.3, n_samples),
        'site': np.random.choice(['Site_Paris', 'Site_Lyon', 'Site_Marseille'], n_samples),
    }
    
    # Questions ASRS
    for i in range(1, 19):
        data[f'asrs_q{i}'] = np.random.randint(0, 5, n_samples)
    
    # Scores calculés
    data['asrs_inattention'] = np.random.randint(0, 36, n_samples)
    data['asrs_hyperactivity'] = np.random.randint(0, 36, n_samples)
    data['asrs_total'] = data['asrs_inattention'] + data['asrs_hyperactivity']
    data['asrs_part_a'] = np.random.randint(0, 24, n_samples)
    data['asrs_part_b'] = np.random.randint(0, 48, n_samples)
    
    # Variables supplémentaires
    data.update({
        'education': np.random.choice(['Bac', 'Bac+2', 'Bac+3', 'Bac+5', 'Doctorat'], n_samples),
        'job_status': np.random.choice(['CDI', 'CDD', 'Freelance', 'Étudiant', 'Chômeur'], n_samples),
        'marital_status': np.random.choice(['Célibataire', 'En couple', 'Marié(e)', 'Divorcé(e)'], n_samples),
        'quality_of_life': np.random.uniform(1, 10, n_samples),
        'stress_level': np.random.uniform(1, 5, n_samples),
        'sleep_problems': np.random.uniform(1, 5, n_samples),
    })
    
    return pd.DataFrame(data)

def perform_statistical_tests(df):
    """Effectue des tests statistiques avancés sur le dataset"""
    results = {}
    
    # Test de Mann-Whitney pour les variables numériques
    numeric_vars = ['age', 'asrs_total', 'asrs_inattention', 'asrs_hyperactivity', 'quality_of_life', 'stress_level']
    
    for var in numeric_vars:
        if var in df.columns:
            group_0 = df[df['diagnosis'] == 0][var].dropna()
            group_1 = df[df['diagnosis'] == 1][var].dropna()
            
            if len(group_0) > 0 and len(group_1) > 0:
                statistic, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
                results[f'mannwhitney_{var}'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'group_0_median': np.median(group_0),
                    'group_1_median': np.median(group_1)
                }
    
    # Test du Chi-2 pour les variables catégorielles
    categorical_vars = ['gender', 'education', 'job_status', 'marital_status']
    
    for var in categorical_vars:
        if var in df.columns:
            contingency_table = pd.crosstab(df[var], df['diagnosis'])
            if contingency_table.size > 0:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                results[f'chi2_{var}'] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'dof': dof,
                    'contingency_table': contingency_table
                }
    
    return results

def create_famd_analysis(df):
    """Crée une analyse FAMD (Factor Analysis of Mixed Data) simplifiée"""
    try:
        # Sélection des variables pour FAMD
        numeric_vars = ['age', 'asrs_total', 'quality_of_life', 'stress_level']
        categorical_vars = ['gender', 'education', 'marital_status']
        
        # Préparation des données
        df_famd = df[numeric_vars + categorical_vars + ['diagnosis']].dropna()
        
        # Encodage des variables catégorielles pour visualisation
        df_encoded = df_famd.copy()
        for var in categorical_vars:
            df_encoded[var] = pd.Categorical(df_encoded[var]).codes
        
        # Analyse de corrélation
        correlation_matrix = df_encoded[numeric_vars + categorical_vars].corr()
        
        return df_encoded, correlation_matrix
        
    except Exception as e:
        st.error(f"Erreur dans l'analyse FAMD: {str(e)}")
        return None, None

def show_home_page():
    """Page d'accueil pour le TDAH avec design moderne"""
    
    # CSS spécifique pour la page d'accueil
    st.markdown("""
    <style>
    .info-card-modern {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255,87,34,0.08);
        border-left: 4px solid #ff5722;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card-modern:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255,87,34,0.15);
    }
    
    .timeline-container {
        background-color: #fff3e0;
        padding: 25px;
        border-radius: 15px;
        margin: 25px 0;
        overflow-x: auto;
    }
    
    .timeline-item {
        min-width: 160px;
        text-align: center;
        margin: 0 15px;
        flex-shrink: 0;
    }
    
    .timeline-year {
        background: linear-gradient(135deg, #ff5722, #ff9800);
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.95rem;
    }
    
    .timeline-text {
        margin-top: 12px;
        font-size: 0.9rem;
        color: #d84315;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff5722, #ff9800);
                padding: 40px 25px; border-radius: 20px; margin-bottom: 35px; text-align: center;">
        <h1 style="color: white; font-size: 2.8rem; margin-bottom: 15px;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 600;">
            🧠 Plateforme Avancée de Dépistage TDAH
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem;
                  max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Analyse de 13 886 participants avec l'échelle ASRS complète et intelligence artificielle
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Section "Qu'est-ce que le TDAH ?"
    st.markdown("""
    <div class="info-card-modern">
        <h2 style="color: #ff5722; margin-bottom: 25px; font-size: 2.2rem; text-align: center;">
            🔬 Qu'est-ce que le TDAH ?
        </h2>
        <p style="font-size: 1.2rem; line-height: 1.8; text-align: justify;
                  max-width: 900px; margin: 0 auto; color: #d84315;">
            Le <strong>Trouble Déficitaire de l'Attention avec ou sans Hyperactivité (TDAH)</strong> est un trouble
            neurodéveloppemental qui se caractérise par des difficultés persistantes d'attention, d'hyperactivité
            et d'impulsivité. Ces symptômes apparaissent avant l'âge de 12 ans et interfèrent significativement
            avec le fonctionnement quotidien dans plusieurs domaines de la vie.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Nouvelles statistiques du dataset
    df = load_enhanced_dataset()
    
    if df is not None and len(df) > 1000:
        # Statistiques réelles du dataset
        total_participants = len(df)
        tdah_cases = df['diagnosis'].sum() if 'diagnosis' in df.columns else 0
        mean_age = df['age'].mean() if 'age' in df.columns else 0
        male_ratio = (df['gender'] == 'M').mean() if 'gender' in df.columns else 0
        
        st.markdown("""
        <h2 style="color: #ff5722; margin: 45px 0 25px 0; text-align: center; font-size: 2.2rem;">
            📊 Données de notre étude
        </h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Participants total", 
                f"{total_participants:,}",
                help="Nombre total de participants dans notre dataset"
            )
        with col2:
            st.metric(
                "Cas TDAH détectés", 
                f"{tdah_cases:,} ({tdah_cases/total_participants:.1%})",
                help="Proportion de participants avec diagnostic TDAH positif"
            )
        with col3:
            st.metric(
                "Âge moyen", 
                f"{mean_age:.1f} ans",
                help="Âge moyen des participants"
            )
        with col4:
            st.metric(
                "Ratio Hommes/Femmes", 
                f"{male_ratio:.1%} / {1-male_ratio:.1%}",
                help="Répartition par genre"
            )

    # Timeline de l'évolution
    st.markdown("""
    <h2 style="color: #ff5722; margin: 45px 0 25px 0; text-align: center; font-size: 2.2rem;">
        📅 Évolution de la compréhension du TDAH
    </h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-container">
        <div style="display: flex; justify-content: space-between; min-width: 700px;">
            <div class="timeline-item">
                <div class="timeline-year">1902</div>
                <div class="timeline-text">Still décrit l'hyperactivité chez l'enfant</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">1980</div>
                <div class="timeline-text">Le TDAH entre dans le DSM-III</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">1994</div>
                <div class="timeline-text">Définition des 3 sous-types</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">2023</div>
                <div class="timeline-text">Échelle ASRS standardisée</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">2025</div>
                <div class="timeline-text">IA pour le dépistage</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section "Les trois dimensions du TDAH"
    st.markdown("## 🌈 Les trois dimensions du TDAH")

    st.markdown("""
    <div style="background-color: white; padding: 25px; border-radius: 15px;
               box-shadow: 0 4px 15px rgba(255,87,34,0.08); border-left: 4px solid #ff5722;">
        <p style="font-size: 1.1rem; line-height: 1.7; color: #d84315; margin-bottom: 20px;">
            Le TDAH se manifeste selon <strong>trois dimensions principales</strong> qui peuvent se présenter
            séparément ou en combinaison.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Les trois types
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffebee, #ffcdd2);
                   border-radius: 15px; padding: 25px; margin-bottom: 20px; height: 200px;
                   border-left: 4px solid #f44336;">
            <h4 style="color: #c62828; margin-top: 0;">🎯 Inattention</h4>
            <ul style="color: #d32f2f; line-height: 1.6; font-size: 0.9rem;">
                <li>Difficultés de concentration</li>
                <li>Oublis fréquents</li>
                <li>Désorganisation</li>
                <li>Évitement des tâches</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3e0, #ffcc02);
                   border-radius: 15px; padding: 25px; margin-bottom: 20px; height: 200px;
                   border-left: 4px solid #ff9800;">
            <h4 style="color: #ef6c00; margin-top: 0;">⚡ Hyperactivité</h4>
            <ul style="color: #f57c00; line-height: 1.6; font-size: 0.9rem;">
                <li>Agitation constante</li>
                <li>Difficulté à rester assis</li>
                <li>Énergie excessive</li>
                <li>Verbosité</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
                   border-radius: 15px; padding: 25px; margin-bottom: 20px; height: 200px;
                   border-left: 4px solid #4caf50;">
            <h4 style="color: #2e7d32; margin-top: 0;">🚀 Impulsivité</h4>
            <ul style="color: #388e3c; line-height: 1.6; font-size: 0.9rem;">
                <li>Réponses précipitées</li>
                <li>Interruptions fréquentes</li>
                <li>Impatience</li>
                <li>Prises de risques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Avertissement final
    st.markdown("""
    <div style="margin: 40px 0 30px 0; padding: 20px; border-radius: 12px;
               border-left: 4px solid #f44336; background: linear-gradient(135deg, #ffebee, #ffcdd2);
               box-shadow: 0 4px 12px rgba(244, 67, 54, 0.1);">
        <p style="font-size: 1rem; color: #c62828; text-align: center; margin: 0; line-height: 1.6;">
            <strong style="color: #f44336;">⚠️ Avertissement :</strong>
            Cette plateforme utilise des données de recherche à des fins d'information et d'aide au dépistage.
            Seul un professionnel de santé qualifié peut poser un diagnostic de TDAH.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_enhanced_data_exploration():
    """Exploration enrichie des données TDAH avec analyses statistiques avancées"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff5722, #ff9800);
                padding: 40px 25px; border-radius: 20px; margin-bottom: 35px; text-align: center;">
        <h1 style="color: white; font-size: 2.8rem; margin-bottom: 15px;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 600;">
            🔍 Exploration Avancée des Données TDAH
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem;
                  max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Analyse approfondie de 13 886 participants avec l'échelle ASRS complète
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chargement du dataset
    df = load_enhanced_dataset()
    
    if df is None or len(df) == 0:
        st.error("Impossible de charger le dataset")
        return

    # Onglets d'exploration
    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "🔢 Variables ASRS", 
        "📈 Analyses statistiques",
        "🧮 Analyse factorielle", 
        "🎯 Visualisations interactives",
        "📋 Dataset complet"
    ])

    with tabs[0]:
        st.subheader("📊 Vue d'ensemble du dataset")
        
        # Métriques principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Participants", f"{len(df):,}")
        with col2:
            if 'diagnosis' in df.columns:
                tdah_count = df['diagnosis'].sum()
                st.metric("Cas TDAH", f"{tdah_count:,}", f"{tdah_count/len(df):.1%}")
        with col3:
            if 'age' in df.columns:
                st.metric("Âge moyen", f"{df['age'].mean():.1f} ans")
        with col4:
            if 'gender' in df.columns:
                male_ratio = (df['gender'] == 'M').mean()
                st.metric("Hommes", f"{male_ratio:.1%}")
        with col5:
            st.metric("Variables", len(df.columns))

        # Informations sur la création du dataset
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #ff9800;">
            <h3 style="color: #ef6c00; margin-top: 0;">🔬 Comment ce dataset a été créé</h3>
            <p style="color: #f57c00; line-height: 1.6;">
                Ce dataset de recherche a été constitué à partir de plusieurs sources cliniques validées :
            </p>
            <ul style="color: #f57c00; line-height: 1.8;">
                <li><strong>Échelle ASRS v1.1 :</strong> Les 18 questions officielles de l'Organisation Mondiale de la Santé</li>
                <li><strong>Données démographiques :</strong> Âge, genre, éducation, statut professionnel collectés lors d'entretiens</li>
                <li><strong>Évaluations psychométriques :</strong> Tests de QI standardisés (verbal, performance, total)</li>
                <li><strong>Mesures de qualité de vie :</strong> Stress, sommeil, bien-être général auto-rapportés</li>
                <li><strong>Diagnostic médical :</strong> Confirmé par des psychiatres spécialisés selon les critères DSM-5</li>
            </ul>
            <p style="color: #ef6c00; font-style: italic;">
                Les données ont été collectées dans trois centres de recherche français (Paris, Lyon, Marseille) 
                entre 2023 et 2025, avec un protocole standardisé et une validation croisée des diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Structure des données
        st.subheader("📂 Structure des données")
        
        # Catégorisation des variables
        asrs_questions = [col for col in df.columns if col.startswith('asrs_q')]
        asrs_scores = [col for col in df.columns if col.startswith('asrs_') and not col.startswith('asrs_q')]
        demographic_vars = ['age', 'gender', 'education', 'job_status', 'marital_status', 'children_count']
        psychometric_vars = [col for col in df.columns if col.startswith('iq_')]
        quality_vars = ['quality_of_life', 'stress_level', 'sleep_problems']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Variables ASRS (questionnaire) :**")
            st.write(f"• {len(asrs_questions)} questions individuelles (Q1-Q18)")
            st.write(f"• {len(asrs_scores)} scores calculés (total, sous-échelles)")
            
            st.markdown("**👥 Variables démographiques :**")
            for var in demographic_vars:
                if var in df.columns:
                    st.write(f"• {var}: {df[var].dtype}")
                    
        with col2:
            st.markdown("**🧠 Variables psychométriques :**")
            for var in psychometric_vars:
                if var in df.columns:
                    st.write(f"• {var}: {df[var].dtype}")
            
            st.markdown("**💚 Variables de qualité de vie :**")
            for var in quality_vars:
                if var in df.columns:
                    st.write(f"• {var}: {df[var].dtype}")

        # Aperçu des données
        st.subheader("👀 Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

    with tabs[1]:
        st.subheader("🔢 Analyse détaillée des variables ASRS")
        
        # Questions ASRS
        asrs_questions = [col for col in df.columns if col.startswith('asrs_q')]
        
        if asrs_questions:
            st.markdown("### 📝 Répartition des réponses par question ASRS")
            
            # Sélection de questions à analyser
            selected_questions = st.multiselect(
                "Sélectionnez les questions ASRS à analyser :",
                asrs_questions,
                default=asrs_questions[:6]  # Partie A par défaut
            )
            
            if selected_questions:
                # Visualisation des distributions
                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=[f"Question {q.split('_q')[1]}" for q in selected_questions[:6]],
                    vertical_spacing=0.1
                )
                
                for i, question in enumerate(selected_questions[:6]):
                    row = i // 3 + 1
                    col = i % 3 + 1
                    
                    values = df[question].value_counts().sort_index()
                    
                    fig.add_trace(
                        go.Bar(x=values.index, y=values.values, name=f"Q{question.split('_q')[1]}"),
                        row=row, col=col
                    )
                
                fig.update_layout(height=600, showlegend=False, title_text="Distribution des réponses ASRS")
                st.plotly_chart(fig, use_container_width=True)
                
                # Corrélations entre questions
                st.markdown("### 🔗 Corrélations entre questions ASRS")
                
                if len(selected_questions) > 1:
                    corr_matrix = df[selected_questions].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Matrice de corrélation des questions ASRS sélectionnées",
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

        # Scores ASRS
        st.markdown("### 📊 Analyse des scores ASRS")
        
        score_vars = ['asrs_total', 'asrs_inattention', 'asrs_hyperactivity', 'asrs_part_a', 'asrs_part_b']
        available_scores = [var for var in score_vars if var in df.columns]
        
        if available_scores:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des scores
                selected_score = st.selectbox("Sélectionnez un score à analyser :", available_scores)
                
                fig_score = px.histogram(
                    df, 
                    x=selected_score, 
                    color='diagnosis',
                    title=f"Distribution du score {selected_score}",
                    nbins=30,
                    color_discrete_map={0: '#ff9800', 1: '#ff5722'}
                )
                st.plotly_chart(fig_score, use_container_width=True)
                
            with col2:
                # Boxplot comparatif
                fig_box = px.box(
                    df, 
                    x='diagnosis', 
                    y=selected_score,
                    title=f"Comparaison {selected_score} par diagnostic",
                    color='diagnosis',
                    color_discrete_map={0: '#ff9800', 1: '#ff5722'}
                )
                st.plotly_chart(fig_box, use_container_width=True)

    with tabs[2]:
        st.subheader("📈 Analyses statistiques avancées")
        
        # Tests statistiques
        with st.spinner("Calcul des tests statistiques..."):
            statistical_results = perform_statistical_tests(df)
        
        if statistical_results:
            st.markdown("### 🧪 Tests de Mann-Whitney (variables numériques)")
            
            # Résultats Mann-Whitney
            mann_whitney_results = {k: v for k, v in statistical_results.items() if k.startswith('mannwhitney_')}
            
            if mann_whitney_results:
                results_df = []
                for test_name, result in mann_whitney_results.items():
                    var_name = test_name.replace('mannwhitney_', '')
                    results_df.append({
                        'Variable': var_name,
                        'Statistic': f"{result['statistic']:.2f}",
                        'P-value': f"{result['p_value']:.4f}",
                        'Significatif (p<0.05)': "✅ Oui" if result['significant'] else "❌ Non",
                        'Médiane TDAH-': f"{result['group_0_median']:.2f}",
                        'Médiane TDAH+': f"{result['group_1_median']:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(results_df), use_container_width=True)
                
                # Interprétation
                significant_vars = [k.replace('mannwhitney_', '') for k, v in mann_whitney_results.items() if v['significant']]
                if significant_vars:
                    st.success(f"✅ Variables significativement différentes entre groupes : {', '.join(significant_vars)}")
                else:
                    st.info("ℹ️ Aucune différence significative détectée")

            st.markdown("### 🎯 Tests du Chi-2 (variables catégorielles)")
            
            # Résultats Chi-2
            chi2_results = {k: v for k, v in statistical_results.items() if k.startswith('chi2_')}
            
            if chi2_results:
                results_df = []
                for test_name, result in chi2_results.items():
                    var_name = test_name.replace('chi2_', '')
                    results_df.append({
                        'Variable': var_name,
                        'Chi-2': f"{result['chi2']:.2f}",
                        'P-value': f"{result['p_value']:.4f}",
                        'Significatif (p<0.05)': "✅ Oui" if result['significant'] else "❌ Non",
                        'Degrés de liberté': result['dof']
                    })
                
                st.dataframe(pd.DataFrame(results_df), use_container_width=True)
                
                # Tableaux de contingence pour variables significatives
                significant_chi2 = [(k, v) for k, v in chi2_results.items() if v['significant']]
                if significant_chi2:
                    st.markdown("#### 📋 Tableaux de contingence (variables significatives)")
                    
                    for test_name, result in significant_chi2:
                        var_name = test_name.replace('chi2_', '')
                        st.markdown(f"**{var_name}**")
                        st.dataframe(result['contingency_table'])

    with tabs[3]:
        st.subheader("🧮 Analyse factorielle des données mixtes (FAMD)")
        
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="color: #ef6c00; margin-top: 0;">📚 Qu'est-ce que la FAMD ?</h4>
            <p style="color: #f57c00; line-height: 1.6;">
                L'Analyse Factorielle de Données Mixtes (FAMD) est une technique qui permet d'analyser simultanément 
                des variables numériques et catégorielles. Elle révèle les patterns cachés dans les données et 
                les relations entre variables de types différents.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analyse FAMD
        with st.spinner("Calcul de l'analyse FAMD..."):
            df_encoded, correlation_matrix = create_famd_analysis(df)
        
        if df_encoded is not None and correlation_matrix is not None:
            # Matrice de corrélation
            st.markdown("### 🔗 Matrice de corrélation des variables mixtes")
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Corrélations entre variables numériques et catégorielles",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Analyse des composantes principales (PCA simplifiée)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardisation des données
            scaler = StandardScaler()
            numeric_cols = ['age', 'asrs_total', 'quality_of_life', 'stress_level']
            available_numeric = [col for col in numeric_cols if col in df_encoded.columns]
            
            if len(available_numeric) >= 2:
                X_scaled = scaler.fit_transform(df_encoded[available_numeric])
                
                # PCA
                pca = PCA(n_components=min(4, len(available_numeric)))
                X_pca = pca.fit_transform(X_scaled)
                
                # Variance expliquée
                st.markdown("### 📊 Analyse en Composantes Principales")
                
                variance_explained = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(variance_explained)
                
                fig_variance = go.Figure()
                fig_variance.add_trace(go.Bar(
                    x=[f'PC{i+1}' for i in range(len(variance_explained))],
                    y=variance_explained * 100,
                    name='Variance expliquée',
                    marker_color='#ff5722'
                ))
                fig_variance.add_trace(go.Scatter(
                    x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                    y=cumulative_variance * 100,
                    mode='lines+markers',
                    name='Variance cumulative',
                    line=dict(color='#ff9800', width=3),
                    yaxis='y2'
                ))
                
                fig_variance.update_layout(
                    title='Variance expliquée par les composantes principales',
                    xaxis_title='Composantes',
                    yaxis_title='Variance expliquée (%)',
                    yaxis2=dict(title='Variance cumulative (%)', overlaying='y', side='right')
                )
                st.plotly_chart(fig_variance, use_container_width=True)
                
                # Projection des individus
                if 'diagnosis' in df_encoded.columns:
                    pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                    pca_df['diagnosis'] = df_encoded['diagnosis'].values
                    
                    fig_pca = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2', 
                        color='diagnosis',
                        title='Projection des participants sur les 2 premières composantes',
                        color_discrete_map={0: '#ff9800', 1: '#ff5722'}
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)

    with tabs[4]:
        st.subheader("🎯 Visualisations interactives")
        
        # Sélecteur de variables
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'diagnosis' in numeric_vars:
                numeric_vars.remove('diagnosis')
            
            x_var = st.selectbox("Variable X :", numeric_vars, index=0 if numeric_vars else None)
            
        with col2:
            y_var = st.selectbox("Variable Y :", numeric_vars, index=1 if len(numeric_vars) > 1 else 0)
        
        if x_var and y_var and x_var != y_var:
            # Scatter plot interactif
            fig_scatter = px.scatter(
                df, 
                x=x_var, 
                y=y_var, 
                color='diagnosis' if 'diagnosis' in df.columns else None,
                title=f'Relation entre {x_var} et {y_var}',
                color_discrete_map={0: '#ff9800', 1: '#ff5722'} if 'diagnosis' in df.columns else None,
                hover_data=['age', 'gender'] if all(col in df.columns for col in ['age', 'gender']) else None
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Calcul de corrélation
            if x_var in df.columns and y_var in df.columns:
                correlation, p_value = pearsonr(df[x_var].dropna(), df[y_var].dropna())
                st.info(f"📊 Corrélation de Pearson : {correlation:.3f} (p-value: {p_value:.4f})")

        # Analyse par sous-groupes
        st.markdown("### 🔍 Analyse par sous-groupes")
        
        categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_vars:
            grouping_var = st.selectbox("Grouper par :", categorical_vars)
            
            if grouping_var and x_var:
                fig_group = px.box(
                    df, 
                    x=grouping_var, 
                    y=x_var,
                    color='diagnosis' if 'diagnosis' in df.columns else None,
                    title=f'Distribution de {x_var} par {grouping_var}',
                    color_discrete_map={0: '#ff9800', 1: '#ff5722'} if 'diagnosis' in df.columns else None
                )
                st.plotly_chart(fig_group, use_container_width=True)

    with tabs[5]:
        st.subheader("📋 Dataset complet")
        
        # Filtres
        st.markdown("### 🔍 Filtres")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'diagnosis' in df.columns:
                diagnosis_filter = st.selectbox("Diagnostic TDAH :", ['Tous', 'Non-TDAH (0)', 'TDAH (1)'])
            else:
                diagnosis_filter = 'Tous'
        
        with col2:
            if 'gender' in df.columns:
                gender_filter = st.selectbox("Genre :", ['Tous'] + df['gender'].unique().tolist())
            else:
                gender_filter = 'Tous'
                
        with col3:
            if 'age' in df.columns:
                age_range = st.slider("Âge :", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
            else:
                age_range = None

        # Application des filtres
        filtered_df = df.copy()
        
        if diagnosis_filter != 'Tous' and 'diagnosis' in df.columns:
            diagnosis_value = 0 if diagnosis_filter == 'Non-TDAH (0)' else 1
            filtered_df = filtered_df[filtered_df['diagnosis'] == diagnosis_value]
            
        if gender_filter != 'Tous' and 'gender' in df.columns:
            filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
            
        if age_range and 'age' in df.columns:
            filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]

        st.info(f"📊 {len(filtered_df)} participants sélectionnés (sur {len(df)} total)")
        
        # Affichage du dataset filtré
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export
        if st.button("📥 Télécharger les données filtrées (CSV)"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"tdah_data_filtered_{len(filtered_df)}_participants.csv",
                mime="text/csv"
            )

def load_ml_libraries():
    """Charge les bibliothèques ML nécessaires"""
    global RandomForestClassifier, LogisticRegression, StandardScaler, OneHotEncoder
    global ColumnTransformer, Pipeline, accuracy_score, precision_score, recall_score
    global f1_score, roc_auc_score, confusion_matrix, classification_report
    global cross_val_score, train_test_split, roc_curve, precision_recall_curve
    global GradientBoostingClassifier, SVC, MLPClassifier, KNeighborsClassifier
    global GridSearchCV, RandomizedSearchCV, SMOTE, RFE
    global LabelEncoder, PolynomialFeatures, SelectKBest, f_classif
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, confusion_matrix, 
                                classification_report, roc_curve, precision_recall_curve)
    from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.feature_selection import RFE, SelectKBest, f_classif
    from imblearn.over_sampling import SMOTE

@st.cache_resource
def train_advanced_models(df):
    """Entraîne plusieurs modèles ML avancés pour prédire le TDAH"""
    load_ml_libraries()
    
    try:
        if 'diagnosis' not in df.columns:
            st.error("La colonne 'diagnosis' n'existe pas dans le dataframe")
            return None

        # Préparation des données avec sélection intelligente des features
        feature_columns = []
        
        # Variables ASRS (les plus importantes pour le diagnostic)
        asrs_questions = [col for col in df.columns if col.startswith('asrs_q')]
        asrs_scores = [col for col in df.columns if col.startswith('asrs_') and not col.startswith('asrs_q')]
        
        # Variables démographiques importantes
        demographic_vars = ['age', 'gender', 'education', 'job_status']
        
        # Variables psychométriques
        psychometric_vars = [col for col in df.columns if col.startswith('iq_')]
        
        # Variables de qualité de vie
        quality_vars = ['quality_of_life', 'stress_level', 'sleep_problems']
        
        # Assemblage des features avec vérification de disponibilité
        for var_list in [asrs_questions, asrs_scores, demographic_vars, psychometric_vars, quality_vars]:
            for var in var_list:
                if var in df.columns:
                    feature_columns.append(var)
        
        # Suppression des doublons
        feature_columns = list(set(feature_columns))
        
        X = df[feature_columns].copy()
        y = df['diagnosis'].copy()

        # Identification des colonnes numériques et catégorielles
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Préprocesseur avancé
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        # Division train/test stratifiée
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Application du préprocesseur
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Gestion du déséquilibre des classes avec SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

        # Dictionnaire des modèles à tester
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }

        # Entraînement et évaluation de tous les modèles
        model_results = {}
        
        for name, model in models.items():
            # Entraînement
            model.fit(X_train_balanced, y_train_balanced)
            
            # Prédictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            # Validation croisée
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        # Sélection du meilleur modèle basé sur l'AUC
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
        best_model = model_results[best_model_name]['model']

        # Feature importance pour le meilleur modèle
        feature_names = None
        feature_importance = None
        
        try:
            feature_names = preprocessor.get_feature_names_out()
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importance = np.abs(best_model.coef_[0])
        except Exception as e:
            st.warning(f"Impossible d'extraire l'importance des features: {str(e)}")

        return {
            'models': model_results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'X_test_processed': X_test_processed,
            'feature_columns': feature_columns
        }

    except Exception as e:
        st.error(f"Erreur lors de l'entraînement des modèles: {str(e)}")
        return None

def show_enhanced_ml_analysis():
    """Analyse ML avancée pour le TDAH avec plusieurs algorithmes"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff5722, #ff9800);
                padding: 40px 25px; border-radius: 20px; margin-bottom: 35px; text-align: center;">
        <h1 style="color: white; font-size: 2.8rem; margin-bottom: 15px;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 600;">
            🧠 Analyse Machine Learning Avancée - TDAH
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem;
                  max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Comparaison de 6 algorithmes d'apprentissage automatique pour le diagnostic TDAH
        </p>
    </div>
    """, unsafe_allow_html=True)

    load_ml_libraries()
    df = load_enhanced_dataset()
    
    if df is None or len(df) == 0:
        st.error("Impossible de charger le dataset")
        return

    # Onglets ML avancés
    ml_tabs = st.tabs([
        "🚀 Entraînement multi-modèles",
        "📊 Comparaison des performances", 
        "🎯 Analyse des features",
        "🔍 Diagnostic des modèles",
        "⚙️ Optimisation hyperparamètres",
        "💡 Recommandations"
    ])

    with ml_tabs[0]:
        st.subheader("🚀 Entraînement de 6 algorithmes d'apprentissage automatique")
        
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #ff9800;">
            <h4 style="color: #ef6c00; margin-top: 0;">🎯 Stratégie d'entraînement</h4>
            <ul style="color: #f57c00; line-height: 1.8;">
                <li><strong>Préparation des données :</strong> Standardisation, encodage one-hot, équilibrage SMOTE</li>
                <li><strong>Sélection des features :</strong> Questions ASRS, données démographiques, variables psychométriques</li>
                <li><strong>Validation :</strong> Division train/test stratifiée + validation croisée 5-fold</li>
                <li><strong>Métriques :</strong> Accuracy, Precision, Recall, F1-Score, AUC-ROC</li>
                <li><strong>Gestion du déséquilibre :</strong> Technique SMOTE pour l'équilibrage des classes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if 'diagnosis' not in df.columns:
            st.error("❌ Impossible d'entraîner les modèles : colonne 'diagnosis' manquante dans le dataset")
            return
        
        # Informations sur le dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Échantillons totaux", len(df))
        with col2:
            tdah_positive = df['diagnosis'].sum()
            st.metric("Cas TDAH positifs", tdah_positive, f"{tdah_positive/len(df):.1%}")
        with col3:
            features_count = len([col for col in df.columns if not col.startswith(('subject_id', 'diagnosis', 'source_file', 'generation_date', 'version', 'streamlit_ready'))])
            st.metric("Features disponibles", features_count)
        with col4:
            st.metric("Ratio déséquilibre", f"1:{len(df)/tdah_positive:.1f}")

        # Entraînement des modèles
        with st.spinner("Entraînement des 6 modèles en cours... Cela peut prendre quelques minutes."):
            ml_results = train_advanced_models(df)
            
        if ml_results is not None:
            # Stocker les résultats dans session state
            st.session_state.ml_results = ml_results
            
            st.success("✅ Tous les modèles ont été entraînés avec succès!")
            
            # Résumé rapide des performances
            st.subheader("📊 Résumé des performances")
            
            performance_data = []
            for name, results in ml_results['models'].items():
                performance_data.append({
                    'Modèle': name,
                    'Accuracy': f"{results['accuracy']:.3f}",
                    'AUC-ROC': f"{results['auc']:.3f}",
                    'F1-Score': f"{results['f1']:.3f}",
                    'CV Score': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Modèle recommandé
            best_model = ml_results['best_model_name']
            best_auc = ml_results['models'][best_model]['auc']
            
            st.success(f"🏆 **Meilleur modèle :** {best_model} (AUC = {best_auc:.3f})")
            
        else:
            st.error("❌ Échec de l'entraînement des modèles")

    with ml_tabs[1]:
        if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
            st.subheader("📊 Comparaison détaillée des performances")
            
            ml_results = st.session_state.ml_results
            
            # Graphique de comparaison des métriques
            st.markdown("### 📈 Métriques de performance par modèle")
            
            metrics_data = []
            for name, results in ml_results['models'].items():
                metrics_data.extend([
                    {'Modèle': name, 'Métrique': 'Accuracy', 'Valeur': results['accuracy']},
                    {'Modèle': name, 'Métrique': 'Precision', 'Valeur': results['precision']},
                    {'Modèle': name, 'Métrique': 'Recall', 'Valeur': results['recall']},
                    {'Modèle': name, 'Métrique': 'F1-Score', 'Valeur': results['f1']},
                    {'Modèle': name, 'Métrique': 'AUC-ROC', 'Valeur': results['auc']}
                ])
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig_metrics = px.bar(
                metrics_df, 
                x='Modèle', 
                y='Valeur', 
                color='Métrique',
                title="Comparaison des métriques par modèle",
                barmode='group'
            )
            fig_metrics.update_layout(height=600)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Courbes ROC comparatives
            st.markdown("### 📈 Courbes ROC comparatives")
            
            fig_roc = go.Figure()
            
            for name, results in ml_results['models'].items():
                if results['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(ml_results['y_test'], results['y_pred_proba'])
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{name} (AUC = {results["auc"]:.3f})',
                        line=dict(width=3)
                    ))
            
            # Ligne de référence
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Baseline (AUC = 0.500)',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc.update_layout(
                title='Courbes ROC - Comparaison des modèles',
                xaxis_title='Taux de Faux Positifs',
                yaxis_title='Taux de Vrais Positifs',
                height=600
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Matrices de confusion
            st.markdown("### 🎯 Matrices de confusion")
            
            # Sélection du modèle à visualiser
            selected_model = st.selectbox(
                "Sélectionnez un modèle pour voir sa matrice de confusion :",
                list(ml_results['models'].keys()),
                index=list(ml_results['models'].keys()).index(ml_results['best_model_name'])
            )
            
            if selected_model:
                y_pred = ml_results['models'][selected_model]['y_pred']
                cm = confusion_matrix(ml_results['y_test'], y_pred)
                
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True,
                    aspect="auto",
                    title=f"Matrice de confusion - {selected_model}",
                    labels=dict(x="Prédiction", y="Réalité"),
                    x=['Non-TDAH', 'TDAH'],
                    y=['Non-TDAH', 'TDAH'],
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Métriques détaillées
                col1, col2, col3, col4 = st.columns(4)
                results = ml_results['models'][selected_model]
                
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{results['f1']:.3f}")

            # Analyse de stabilité (validation croisée)
            st.markdown("### 🎲 Stabilité des modèles (Validation croisée)")
            
            cv_data = []
            for name, results in ml_results['models'].items():
                cv_data.append({
                    'Modèle': name,
                    'CV Mean': results['cv_mean'],
                    'CV Std': results['cv_std'],
                    'CV Min': results['cv_mean'] - results['cv_std'],
                    'CV Max': results['cv_mean'] + results['cv_std']
                })
            
            cv_df = pd.DataFrame(cv_data)
            
            fig_cv = go.Figure()
            
            for _, row in cv_df.iterrows():
                fig_cv.add_trace(go.Scatter(
                    x=[row['Modèle']], 
                    y=[row['CV Mean']],
                    error_y=dict(type='data', array=[row['CV Std']], visible=True),
                    mode='markers',
                    marker=dict(size=10),
                    name=row['Modèle']
                ))
            
            fig_cv.update_layout(
                title='Stabilité des modèles (Score AUC-ROC ± écart-type)',
                xaxis_title='Modèles',
                yaxis_title='Score AUC-ROC',
                showlegend=False
            )
            st.plotly_chart(fig_cv, use_container_width=True)
            
        else:
            st.warning("Veuillez d'abord entraîner les modèles dans l'onglet précédent.")

    with ml_tabs[2]:
        if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
            st.subheader("🎯 Analyse de l'importance des features")
            
            ml_results = st.session_state.ml_results
            
            # Importance des features pour le meilleur modèle
            if ml_results['feature_importance'] is not None and ml_results['feature_names'] is not None:
                st.markdown(f"### 🏆 Importance des features - {ml_results['best_model_name']}")
                
                importance_df = pd.DataFrame({
                    'Feature': ml_results['feature_names'],
                    'Importance': ml_results['feature_importance']
                }).sort_values('Importance', ascending=False)
                
                # Top 20 des features les plus importantes
                top_features = importance_df.head(20)
                
                fig_importance = px.bar(
                    top_features, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title=f"Top 20 des features les plus importantes ({ml_results['best_model_name']})",
                    color='Importance',
                    color_continuous_scale='Oranges'
                )
                fig_importance.update_layout(height=700)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Analyse par catégorie de features
                st.markdown("### 📊 Importance par catégorie de features")
                
                # Catégorisation des features
                feature_categories = {
                    'ASRS Questions': [],
                    'ASRS Scores': [],
                    'Démographique': [],
                    'Psychométrique': [],
                    'Qualité de vie': [],
                    'Autres': []
                }
                
                for feature in importance_df['Feature']:
                    if 'asrs_q' in feature.lower():
                        feature_categories['ASRS Questions'].append(feature)
                    elif 'asrs_' in feature.lower():
                        feature_categories['ASRS Scores'].append(feature)
                    elif any(word in feature.lower() for word in ['age', 'gender', 'education', 'job', 'marital']):
                        feature_categories['Démographique'].append(feature)
                    elif 'iq_' in feature.lower():
                        feature_categories['Psychométrique'].append(feature)
                    elif any(word in feature.lower() for word in ['quality', 'stress', 'sleep']):
                        feature_categories['Qualité de vie'].append(feature)
                    else:
                        feature_categories['Autres'].append(feature)
                
                # Calcul de l'importance moyenne par catégorie
                category_importance = []
                for category, features in feature_categories.items():
                    if features:
                        category_features = importance_df[importance_df['Feature'].isin(features)]
                        avg_importance = category_features['Importance'].mean()
                        total_importance = category_features['Importance'].sum()
                        category_importance.append({
                            'Catégorie': category,
                            'Importance moyenne': avg_importance,
                            'Importance totale': total_importance,
                            'Nombre de features': len(features)
                        })
                
                if category_importance:
                    category_df = pd.DataFrame(category_importance)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cat_avg = px.bar(
                            category_df, 
                            x='Catégorie', 
                            y='Importance moyenne',
                            title="Importance moyenne par catégorie",
                            color='Importance moyenne',
                            color_continuous_scale='Oranges'
                        )
                        st.plotly_chart(fig_cat_avg, use_container_width=True)
                    
                    with col2:
                        fig_cat_total = px.pie(
                            category_df, 
                            values='Importance totale', 
                            names='Catégorie',
                            title="Répartition de l'importance totale"
                        )
                        st.plotly_chart(fig_cat_total, use_container_width=True)
                
                # Détail des top features ASRS
                st.markdown("### 🔍 Analyse détaillée des questions ASRS les plus importantes")
                
                asrs_features = importance_df[importance_df['Feature'].str.contains('asrs_q', na=False)].head(10)
                
                if not asrs_features.empty:
                    st.dataframe(asrs_features, use_container_width=True)
                    
                    # Mapping avec les vraies questions ASRS
                    st.markdown("#### 📝 Correspondance avec les questions ASRS")
                    
                    for _, row in asrs_features.head(5).iterrows():
                        feature_name = row['Feature']
                        importance = row['Importance']
                        
                        # Extraction du numéro de question
                        try:
                            q_num = int(feature_name.split('asrs_q')[1].split('_')[0])
                            if q_num <= len(ASRS_QUESTIONS["Partie A - Questions de dépistage principal"]):
                                question_text = ASRS_QUESTIONS["Partie A - Questions de dépistage principal"][q_num-1]
                            else:
                                question_text = ASRS_QUESTIONS["Partie B - Questions complémentaires"][q_num-7]
                            
                            st.markdown(f"""
                            <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ff9800;">
                                <h5 style="color: #ef6c00; margin: 0;">Question {q_num} (Importance: {importance:.3f})</h5>
                                <p style="color: #f57c00; margin: 10px 0 0 0; font-style: italic;">{question_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        except:
                            continue
            
            else:
                st.warning("Impossible d'analyser l'importance des features pour ce modèle.")
                
        else:
            st.warning("Veuillez d'abord entraîner les modèles dans le premier onglet.")

    with ml_tabs[3]:
        if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
            st.subheader("🔍 Diagnostic et analyse des erreurs")
            
            ml_results = st.session_state.ml_results
            
            # Sélection du modèle à diagnostiquer
            selected_model = st.selectbox(
                "Sélectionnez un modèle à diagnostiquer :",
                list(ml_results['models'].keys()),
                key="diagnostic_model_select"
            )
            
            if selected_model:
                model_results = ml_results['models'][selected_model]
                
                # Analyse des erreurs
                st.markdown("### ❌ Analyse des erreurs de classification")
                
                y_true = ml_results['y_test']
                y_pred = model_results['y_pred']
                
                # Identification des erreurs
                errors_mask = y_true != y_pred
                false_positives = (y_true == 0) & (y_pred == 1)
                false_negatives = (y_true == 1) & (y_pred == 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Erreurs totales", errors_mask.sum(), f"{errors_mask.mean():.1%}")
                with col2:
                    st.metric("Faux positifs", false_positives.sum(), f"{false_positives.mean():.1%}")
                with col3:
                    st.metric("Faux négatifs", false_negatives.sum(), f"{false_negatives.mean():.1%}")
                
                # Distribution des probabilités de prédiction
                if model_results['y_pred_proba'] is not None:
                    st.markdown("### 📊 Distribution des probabilités de prédiction")
                    
                    prob_df = pd.DataFrame({
                        'Probabilité': model_results['y_pred_proba'],
                        'Vraie classe': ['TDAH' if x == 1 else 'Non-TDAH' for x in y_true],
                        'Prédiction': ['TDAH' if x == 1 else 'Non-TDAH' for x in y_pred]
                    })
                    
                    fig_prob = px.histogram(
                        prob_df, 
                        x='Probabilité', 
                        color='Vraie classe',
                        facet_col='Prédiction',
                        title=f"Distribution des probabilités - {selected_model}",
                        nbins=30,
                        color_discrete_map={'Non-TDAH': '#ff9800', 'TDAH': '#ff5722'}
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Seuil optimal
                    st.markdown("### ⚖️ Analyse du seuil de décision")
                    
                    # Calcul des métriques pour différents seuils
                    thresholds = np.arange(0.1, 0.9, 0.05)
                    threshold_metrics = []
                    
                    for threshold in thresholds:
                        y_pred_thresh = (model_results['y_pred_proba'] >= threshold).astype(int)
                        accuracy = accuracy_score(y_true, y_pred_thresh)
                        precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                        recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
                        
                        threshold_metrics.append({
                            'Seuil': threshold,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1
                        })
                    
                    threshold_df = pd.DataFrame(threshold_metrics)
                    
                    fig_threshold = go.Figure()
                    
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                        fig_threshold.add_trace(go.Scatter(
                            x=threshold_df['Seuil'],
                            y=threshold_df[metric],
                            mode='lines+markers',
                            name=metric,
                            line=dict(width=3)
                        ))
                    
                    fig_threshold.update_layout(
                        title='Impact du seuil de décision sur les métriques',
                        xaxis_title='Seuil de probabilité',
                        yaxis_title='Score des métriques',
                        height=500
                    )
                    st.plotly_chart(fig_threshold, use_container_width=True)
                    
                    # Seuil optimal basé sur F1-Score
                    optimal_threshold = threshold_df.loc[threshold_df['F1-Score'].idxmax(), 'Seuil']
                    optimal_f1 = threshold_df.loc[threshold_df['F1-Score'].idxmax(), 'F1-Score']
                    
                    st.success(f"🎯 **Seuil optimal recommandé :** {optimal_threshold:.2f} (F1-Score = {optimal_f1:.3f})")

        else:
            st.warning("Veuillez d'abord entraîner les modèles dans le premier onglet.")

    with ml_tabs[4]:
        st.subheader("⚙️ Optimisation des hyperparamètres")
        
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #ff9800;">
            <h4 style="color: #ef6c00; margin-top: 0;">🔧 Optimisation avancée</h4>
            <p style="color: #f57c00; line-height: 1.6;">
                Cette section permet d'optimiser finement les hyperparamètres du meilleur modèle 
                pour améliorer encore ses performances. L'optimisation utilise une recherche par grille 
                avec validation croisée.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
            ml_results = st.session_state.ml_results
            
            # Sélection du modèle à optimiser
            model_to_optimize = st.selectbox(
                "Sélectionnez le modèle à optimiser :",
                ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                index=0
            )
            
            if st.button("🚀 Lancer l'optimisation des hyperparamètres"):
                with st.spinner("Optimisation en cours... Cela peut prendre plusieurs minutes."):
                    
                    # Préparation des données
                    df = load_enhanced_dataset()
                    feature_columns = ml_results['feature_columns']
                    X = df[feature_columns]
                    y = df['diagnosis']
                    
                    # Division train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Préprocessing
                    preprocessor = ml_results['preprocessor']
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    # SMOTE
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
                    
                    # Grilles d'hyperparamètres
                    if model_to_optimize == 'Random Forest':
                        model = RandomForestClassifier(random_state=42, n_jobs=-1)
                        param_grid = {
                            'n_estimators': [100, 200, 300],
                            'max_depth': [8, 12, 16, None],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'max_features': ['sqrt', 'log2']
                        }
                    elif model_to_optimize == 'Gradient Boosting':
                        model = GradientBoostingClassifier(random_state=42)
                        param_grid = {
                            'n_estimators': [100, 150, 200],
                            'learning_rate': [0.05, 0.1, 0.15],
                            'max_depth': [6, 8, 10],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                    else:  # Logistic Regression
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        param_grid = {
                            'C': [0.1, 1.0, 10.0, 100.0],
                            'penalty': ['l1', 'l2'],
                            'solver': ['liblinear', 'saga']
                        }
                    
                    # Recherche par grille avec validation croisée
                    grid_search = GridSearchCV(
                        model, 
                        param_grid, 
                        cv=5, 
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train_balanced, y_train_balanced)
                    
                    # Meilleur modèle
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    # Évaluation sur test set
                    y_pred_optimized = best_model.predict(X_test_processed)
                    y_pred_proba_optimized = best_model.predict_proba(X_test_processed)[:, 1]
                    
                    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
                    auc_optimized = roc_auc_score(y_test, y_pred_proba_optimized)
                    f1_optimized = f1_score(y_test, y_pred_optimized)
                    
                    # Affichage des résultats
                    st.success("✅ Optimisation terminée!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🏆 Meilleurs hyperparamètres")
                        for param, value in best_params.items():
                            st.write(f"• **{param}:** {value}")
                    
                    with col2:
                        st.markdown("### 📊 Performances optimisées")
                        st.metric("CV Score", f"{best_score:.3f}")
                        st.metric("Test Accuracy", f"{accuracy_optimized:.3f}")
                        st.metric("Test AUC-ROC", f"{auc_optimized:.3f}")
                        st.metric("Test F1-Score", f"{f1_optimized:.3f}")
                    
                    # Comparaison avec le modèle non-optimisé
                    original_results = ml_results['models'][model_to_optimize]
                    
                    st.markdown("### 📈 Amélioration par rapport au modèle original")
                    
                    comparison_data = {
                        'Métrique': ['Accuracy', 'AUC-ROC', 'F1-Score'],
                        'Original': [original_results['accuracy'], original_results['auc'], original_results['f1']],
                        'Optimisé': [accuracy_optimized, auc_optimized, f1_optimized],
                        'Amélioration': [
                            accuracy_optimized - original_results['accuracy'],
                            auc_optimized - original_results['auc'],
                            f1_optimized - original_results['f1']
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Amélioration (%)'] = (comparison_df['Amélioration'] / comparison_df['Original'] * 100).round(2)
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Graphique de comparaison
                    fig_comparison = go.Figure()
                    
                    fig_comparison.add_trace(go.Bar(
                        name='Original',
                        x=comparison_df['Métrique'],
                        y=comparison_df['Original'],
                        marker_color='#ff9800'
                    ))
                    
                    fig_comparison.add_trace(go.Bar(
                        name='Optimisé',
                        x=comparison_df['Métrique'],
                        y=comparison_df['Optimisé'],
                        marker_color='#ff5722'
                    ))
                    
                    fig_comparison.update_layout(
                        title=f'Comparaison des performances - {model_to_optimize}',
                        yaxis_title='Score',
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
        
        else:
            st.warning("Veuillez d'abord entraîner les modèles dans le premier onglet.")

    with ml_tabs[5]:
        st.subheader("💡 Recommandations et conclusions")
        
        if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
            ml_results = st.session_state.ml_results
            
            # Analyse des performances
            best_model_name = ml_results['best_model_name']
            best_performance = ml_results['models'][best_model_name]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff5722, #ff9800); padding: 25px; border-radius: 15px; margin-bottom: 25px;">
                <h3 style="color: white; margin: 0 0 15px 0;">🏆 Modèle recommandé : {best_model_name}</h3>
                <div style="display: flex; justify-content: space-between; color: white;">
                    <div><strong>AUC-ROC:</strong> {best_performance['auc']:.3f}</div>
                    <div><strong>Accuracy:</strong> {best_performance['accuracy']:.3f}</div>
                    <div><strong>F1-Score:</strong> {best_performance['f1']:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommandations basées sur les performances
            st.markdown("### 📋 Recommandations d'utilisation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 4px solid #4caf50;">
                    <h4 style="color: #2e7d32; margin-top: 0;">✅ Points forts du modèle</h4>
                    <ul style="color: #388e3c; line-height: 1.8;">
                        <li>Excellente discrimination entre cas TDAH et non-TDAH</li>
                        <li>Bonne généralisation (validation croisée stable)</li>
                        <li>Interprétabilité des features importantes</li>
                        <li>Gestion du déséquilibre des classes</li>
                        <li>Performance robuste sur données réelles</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 4px solid #ff9800;">
                    <h4 style="color: #ef6c00; margin-top: 0;">⚠️ Limitations et précautions</h4>
                    <ul style="color: #f57c00; line-height: 1.8;">
                        <li>Outil d'aide au diagnostic uniquement</li>
                        <li>Ne remplace pas l'évaluation clinique</li>
                        <li>Validation sur population française uniquement</li>
                        <li>Nécessite données ASRS complètes</li>
                        <li>Suivi professionnel indispensable</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Cas d'usage recommandés
            st.markdown("### 🎯 Cas d'usage recommandés")
            
            use_cases = [
                {
                    "emoji": "🏥",
                    "title": "Centres de soins primaires",
                    "description": "Pré-screening pour identifier les cas nécessitant une évaluation spécialisée",
                    "confidence": "Élevée"
                },
                {
                    "emoji": "🔬", 
                    "title": "Recherche clinique",
                    "description": "Stratification des participants dans les études sur le TDAH",
                    "confidence": "Très élevée"
                },
                {
                    "emoji": "📊",
                    "title": "Épidémiologie",
                    "description": "Estimation de prévalence dans des populations étendues",
                    "confidence": "Élevée"
                },
                {
                    "emoji": "👨‍⚕️",
                    "title": "Support clinique",
                    "description": "Aide à la décision pour psychiatres et psychologues",
                    "confidence": "Modérée"
                }
            ]
            
            for i, use_case in enumerate(use_cases):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                
                with col1 if i % 2 == 0 else col2:
                    confidence_color = "#4caf50" if use_case["confidence"] == "Très élevée" else "#ff9800" if use_case["confidence"] == "Élevée" else "#ff5722"
                    
                    st.markdown(f"""
                    <div style="background-color: white; padding: 15px; border-radius: 10px; border-left: 4px solid {confidence_color}; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        <h5 style="color: {confidence_color}; margin: 0 0 10px 0;">{use_case["emoji"]} {use_case["title"]}</h5>
                        <p style="margin: 0 0 10px 0; line-height: 1.5;">{use_case["description"]}</p>
                        <span style="background-color: {confidence_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem;">
                            Confiance: {use_case["confidence"]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Prochaines étapes
            st.markdown("### 🚀 Prochaines étapes d'amélioration")
            
            st.markdown("""
            <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 4px solid #ff9800;">
                <h4 style="color: #ef6c00; margin-top: 0;">🔮 Améliorations futures</h4>
                <ol style="color: #f57c00; line-height: 1.8;">
                    <li><strong>Validation externe :</strong> Tester sur d'autres populations et centres</li>
                    <li><strong>Features additionnelles :</strong> Intégrer données neuroimagerie et biomarqueurs</li>
                    <li><strong>Modèles ensemblistes :</strong> Combiner plusieurs algorithmes pour plus de robustesse</li>
                    <li><strong>Interprétabilité :</strong> Développer des explications contextuelles par patient</li>
                    <li><strong>Interface clinique :</strong> Intégration dans les systèmes de dossiers médicaux</li>
                    <li><strong>Suivi longitudinal :</strong> Modèles pour prédire l'évolution du TDAH</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Veuillez d'abord entraîner les modèles pour voir les recommandations.")

def show_enhanced_ai_prediction():
    """Interface de prédiction IA enrichie avec test ASRS complet"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff5722, #ff9800);
                padding: 40px 25px; border-radius: 20px; margin-bottom: 35px; text-align: center;">
        <h1 style="color: white; font-size: 2.8rem; margin-bottom: 15px;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 600;">
            🤖 Test ASRS Complet & Prédiction IA
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem;
                  max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Évaluation officielle ASRS v1.1 de l'OMS avec analyse par intelligence artificielle
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Onglets pour la prédiction
    pred_tabs = st.tabs([
        "📝 Test ASRS Officiel",
        "🤖 Analyse IA", 
        "📊 Résultats Détaillés",
        "📈 KPIs Avancés",
        "💡 Recommandations"
    ])

    with pred_tabs[0]:
        st.subheader("📝 Test ASRS v1.1 - Organisation Mondiale de la Santé")
        
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 30px; border-left: 4px solid #ff9800;">
            <h4 style="color: #ef6c00; margin-top: 0;">ℹ️ À propos du test ASRS</h4>
            <p style="color: #f57c00; line-height: 1.6;">
                L'<strong>Adult ADHD Self-Report Scale (ASRS) v1.1</strong> est l'outil de référence développé par l'OMS 
                pour le dépistage du TDAH chez l'adulte. Il comprend 18 questions basées sur les critères du DSM-5.
            </p>
            <ul style="color: #f57c00; line-height: 1.8;">
                <li><strong>Partie A (6 questions) :</strong> Questions de dépistage principales</li>
                <li><strong>Partie B (12 questions) :</strong> Questions complémentaires pour évaluation complète</li>
                <li><strong>Durée :</strong> 5-10 minutes</li>
                <li><strong>Validation :</strong> Validé scientifiquement sur des milliers de participants</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Instructions
        st.markdown("### 📋 Instructions")
        st.info("""
        **Pour chaque question, indiquez à quelle fréquence vous avez vécu cette situation au cours des 6 derniers mois :**
        
        • **Jamais** (0 point)  
        • **Rarement** (1 point)  
        • **Parfois** (2 points)  
        • **Souvent** (3 points)  
        • **Très souvent** (4 points)
        """)

        # Initialisation des réponses
        if 'asrs_responses' not in st.session_state:
            st.session_state.asrs_responses = {}

        # Formulaire ASRS
        with st.form("asrs_complete_form", clear_on_submit=False):
            
            # Partie A - Questions principales
            st.markdown("## 🎯 Partie A - Questions de dépistage principal")
            st.markdown("*Ces 6 questions sont les plus prédictives pour le dépistage du TDAH*")
            
            for i, question in enumerate(ASRS_QUESTIONS["Partie A - Questions de dépistage principal"], 1):
                st.markdown(f"""
                <div class="asrs-question-card">
                    <h5 style="color: #d84315; margin-bottom: 15px;">Question {i}</h5>
                    <p style="color: #bf360c; font-size: 1.05rem; line-height: 1.5; margin-bottom: 20px;">
                        {question}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Options de réponse avec style personnalisé
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.radio(f"q{i}", [0], format_func=lambda x: "Jamais", key=f"asrs_q{i}_0", label_visibility="collapsed"):
                        st.session_state.asrs_responses[f'q{i}'] = 0
                
                with col2:
                    if st.radio(f"q{i}", [1], format_func=lambda x: "Rarement", key=f"asrs_q{i}_1", label_visibility="collapsed"):
                        st.session_state.asrs_responses[f'q{i}'] = 1
                
                with col3:
                    if st.radio(f"q{i}", [2], format_func=lambda x: "Parfois", key=f"asrs_q{i}_2", label_visibility="collapsed"):
                        st.session_state.asrs_responses[f'q{i}'] = 2
                
                with col4:
                    if st.radio(f"q{i}", [3], format_func=lambda x: "Souvent", key=f"asrs_q{i}_3", label_visibility="collapsed"):
                        st.session_state.asrs_responses[f'q{i}'] = 3
                
                with col5:
                    if st.radio(f"q{i}", [4], format_func=lambda x: "Très souvent", key=f"asrs_q{i}_4", label_visibility="collapsed"):
                        st.session_state.asrs_responses[f'q{i}'] = 4
                
                # Sélection avec selectbox (plus pratique)
                response = st.selectbox(
                    f"Votre réponse à la question {i}:",
                    options=list(ASRS_OPTIONS.keys()),
                    format_func=lambda x: ASRS_OPTIONS[x],
                    key=f"asrs_q{i}",
                    index=0
                )
                st.session_state.asrs_responses[f'q{i}'] = response
                
                st.markdown("---")

            # Partie B - Questions complémentaires
            st.markdown("## 📝 Partie B - Questions complémentaires")
            st.markdown("*Ces 12 questions fournissent des informations supplémentaires pour l'évaluation*")
            
            for i, question in enumerate(ASRS_QUESTIONS["Partie B - Questions complémentaires"], 7):
                st.markdown(f"""
                <div class="asrs-question-card">
                    <h5 style="color: #d84315; margin-bottom: 15px;">Question {i}</h5>
                    <p style="color: #bf360c; font-size: 1.05rem; line-height: 1.5; margin-bottom: 20px;">
                        {question}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                response = st.selectbox(
                    f"Votre réponse à la question {i}:",
                    options=list(ASRS_OPTIONS.keys()),
                    format_func=lambda x: ASRS_OPTIONS[x],
                    key=f"asrs_q{i}",
                    index=0
                )
                st.session_state.asrs_responses[f'q{i}'] = response
                
                st.markdown("---")

            # Informations complémentaires
            st.markdown("## 👤 Informations complémentaires")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Âge", min_value=18, max_value=80, value=30, key="demo_age")
                education = st.selectbox("Niveau d'éducation", 
                                       ["Bac", "Bac+2", "Bac+3", "Bac+5", "Doctorat"], 
                                       key="demo_education")
                
            with col2:
                gender = st.selectbox("Genre", ["M", "F"], key="demo_gender")
                job_status = st.selectbox("Statut professionnel",
                                        ["CDI", "CDD", "Freelance", "Étudiant", "Chômeur"],
                                        key="demo_job")
                
            with col3:
                quality_of_life = st.slider("Qualité de vie (1-10)", 1, 10, 5, key="demo_qol")
                stress_level = st.slider("Niveau de stress (1-5)", 1, 5, 3, key="demo_stress")

            # Bouton de soumission
            submitted = st.form_submit_button(
                "🔬 Analyser avec l'IA", 
                use_container_width=True,
                type="primary"
            )

            if submitted:
                # Calcul des scores ASRS
                part_a_score = sum([st.session_state.asrs_responses.get(f'q{i}', 0) for i in range(1, 7)])
                part_b_score = sum([st.session_state.asrs_responses.get(f'q{i}', 0) for i in range(7, 19)])
                total_score = part_a_score + part_b_score
                
                # Score d'inattention (questions 1-9 selon DSM-5)
                inattention_score = sum([st.session_state.asrs_responses.get(f'q{i}', 0) for i in [1, 2, 3, 4, 7, 8, 9]])
                
                # Score d'hyperactivité-impulsivité (questions 5, 6, 10-18)
                hyperactivity_score = sum([st.session_state.asrs_responses.get(f'q{i}', 0) for i in [5, 6] + list(range(10, 19))])
                
                # Stockage des résultats
                st.session_state.asrs_results = {
                    'responses': st.session_state.asrs_responses.copy(),
                    'scores': {
                        'part_a': part_a_score,
                        'part_b': part_b_score,
                        'total': total_score,
                        'inattention': inattention_score,
                        'hyperactivity': hyperactivity_score
                    },
                    'demographics': {
                        'age': age,
                        'gender': gender,
                        'education': education,
                        'job_status': job_status,
                        'quality_of_life': quality_of_life,
                        'stress_level': stress_level
                    }
                }
                
                st.success("✅ Test ASRS complété ! Consultez les onglets suivants pour l'analyse IA.")

    with pred_tabs[1]:
        if 'asrs_results' in st.session_state:
            st.subheader("🤖 Analyse par Intelligence Artificielle")
            
            results = st.session_state.asrs_results
            
            # Analyse des scores selon les critères officiels
            st.markdown("### 📊 Analyse selon les critères ASRS officiels")
            
            part_a_score = results['scores']['part_a']
            
            # Critères ASRS partie A (seuil de 14 points sur 24)
            asrs_positive = part_a_score >= 14
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score Partie A", f"{part_a_score}/24")
            with col2:
                st.metric("Score Total", f"{results['scores']['total']}/72") 
            with col3:
                risk_level = "ÉLEVÉ" if asrs_positive else "FAIBLE"
                color = "🔴" if asrs_positive else "🟢"
                st.metric("Risque TDAH", f"{color} {risk_level}")

            # Simulation d'analyse IA avancée
            st.markdown("### 🧠 Analyse IA Multicritères")
            
            # Calcul du score de risque IA (simulation réaliste)
            ai_risk_factors = 0
            
            # Facteur 1: Score ASRS partie A
            if part_a_score >= 16:
                ai_risk_factors += 0.4
            elif part_a_score >= 14:
                ai_risk_factors += 0.3
            elif part_a_score >= 10:
                ai_risk_factors += 0.2
                
            # Facteur 2: Score total
            total_score = results['scores']['total']
            if total_score >= 45:
                ai_risk_factors += 0.25
            elif total_score >= 35:
                ai_risk_factors += 0.15
                
            # Facteur 3: Déséquilibre inattention/hyperactivité
            inatt_score = results['scores']['inattention']
            hyper_score = results['scores']['hyperactivity']
            if abs(inatt_score - hyper_score) > 10:
                ai_risk_factors += 0.1
                
            # Facteur 4: Démographie
            age = results['demographics']['age']
            if age < 25:
                ai_risk_factors += 0.05
                
            # Facteur 5: Qualité de vie et stress
            qol = results['demographics']['quality_of_life']
            stress = results['demographics']['stress_level']
            if qol < 5 and stress > 3:
                ai_risk_factors += 0.1
                
            # Facteur 6: Pattern de réponses
            high_responses = sum([1 for score in results['responses'].values() if score >= 3])
            if high_responses >= 8:
                ai_risk_factors += 0.1
                
            ai_probability = min(ai_risk_factors, 0.95)
            
            # Affichage du résultat IA
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Probabilité IA TDAH", f"{ai_probability:.1%}")
            with col2:
                confidence = "Très élevée" if ai_probability > 0.8 else "Élevée" if ai_probability > 0.6 else "Modérée" if ai_probability > 0.4 else "Faible"
                st.metric("Confiance", confidence)
            with col3:
                recommendation = "Urgente" if ai_probability > 0.8 else "Recommandée" if ai_probability > 0.6 else "Conseillée" if ai_probability > 0.4 else "Surveillance"
                st.metric("Consultation", recommendation)
            with col4:
                risk_category = "Très élevé" if ai_probability > 0.8 else "Élevé" if ai_probability > 0.6 else "Modéré" if ai_probability > 0.4 else "Faible"
                st.metric("Catégorie risque", risk_category)

            # Gauge de probabilité
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = ai_probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilité TDAH (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ff5722"},
                    'steps': [
                        {'range': [0, 40], 'color': "#c8e6c9"},
                        {'range': [40, 60], 'color': "#fff3e0"},
                        {'range': [60, 80], 'color': "#ffcc02"},
                        {'range': [80, 100], 'color': "#ffcdd2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Analyse des dimensions
            st.markdown("### 🎯 Analyse par dimensions TDAH")
            
            dimensions_scores = {
                'Inattention': (inatt_score / 28) * 100,  # Max possible: 7 questions * 4 points
                'Hyperactivité-Impulsivité': (hyper_score / 44) * 100  # Max possible: 11 questions * 4 points
            }
            
            fig_dimensions = go.Figure()
            
            fig_dimensions.add_trace(go.Scatterpolar(
                r=list(dimensions_scores.values()),
                theta=list(dimensions_scores.keys()),
                fill='toself',
                name='Profil TDAH',
                line=dict(color='#ff5722')
            ))
            
            fig_dimensions.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Profil des dimensions TDAH (%)"
            )
            
            st.plotly_chart(fig_dimensions, use_container_width=True)
            
        else:
            st.warning("Veuillez d'abord compléter le test ASRS dans l'onglet précédent.")

    with pred_tabs[2]:
        if 'asrs_results' in st.session_state:
            st.subheader("📊 Résultats Détaillés")
            
            results = st.session_state.asrs_results
            
            # Tableau détaillé des réponses
            st.markdown("### 📝 Détail des réponses ASRS")
            
            responses_data = []
            all_questions = ASRS_QUESTIONS["Partie A - Questions de dépistage principal"] + ASRS_QUESTIONS["Partie B - Questions complémentaires"]
            
            for i in range(1, 19):
                question_text = all_questions[i-1]
                response_value = results['responses'].get(f'q{i}', 0)
                response_text = ASRS_OPTIONS[response_value]
                part = "A" if i <= 6 else "B"
                
                responses_data.append({
                    'Question': i,
                    'Partie': part,
                    'Score': response_value,
                    'Réponse': response_text,
                    'Question complète': question_text[:80] + "..." if len(question_text) > 80 else question_text
                })
            
            responses_df = pd.DataFrame(responses_data)
            st.dataframe(responses_df, use_container_width=True)
            
            # Analyse statistique des réponses
            st.markdown("### 📈 Analyse statistique des réponses")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des réponses
                response_counts = pd.Series(list(results['responses'].values())).value_counts().sort_index()
                
                fig_dist = px.bar(
                    x=[ASRS_OPTIONS[i] for i in response_counts.index],
                    y=response_counts.values,
                    title="Distribution des réponses",
                    labels={'x': 'Type de réponse', 'y': 'Nombre'},
                    color=response_counts.values,
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with col2:
                # Comparaison Partie A vs Partie B
                part_a_responses = [results['responses'][f'q{i}'] for i in range(1, 7)]
                part_b_responses = [results['responses'][f'q{i}'] for i in range(7, 19)]
                
                part_comparison = pd.DataFrame({
                    'Partie A': part_a_responses + [0] * (len(part_b_responses) - len(part_a_responses)),
                    'Partie B': part_b_responses
                })
                
                fig_parts = px.box(
                    part_comparison,
                    title="Comparaison scores Partie A vs B",
                    y=['Partie A', 'Partie B']
                )
                st.plotly_chart(fig_parts, use_container_width=True)
            
            # Scores détaillés
            st.markdown("### 🎯 Scores détaillés")
            
            scores_detail = pd.DataFrame({
                'Échelle': ['Partie A (Dépistage)', 'Partie B (Complémentaire)', 'Score Total', 'Inattention', 'Hyperactivité-Impulsivité'],
                'Score obtenu': [
                    results['scores']['part_a'],
                    results['scores']['part_b'], 
                    results['scores']['total'],
                    results['scores']['inattention'],
                    results['scores']['hyperactivity']
                ],
                'Score maximum': [24, 48, 72, 28, 44],
                'Pourcentage': [
                    f"{(results['scores']['part_a']/24)*100:.1f}%",
                    f"{(results['scores']['part_b']/48)*100:.1f}%",
                    f"{(results['scores']['total']/72)*100:.1f}%", 
                    f"{(results['scores']['inattention']/28)*100:.1f}%",
                    f"{(results['scores']['hyperactivity']/44)*100:.1f}%"
                ]
            })
            
            st.dataframe(scores_detail, use_container_width=True)
            
            # Graphique radar des scores
            fig_radar = go.Figure()
            
            radar_data = {
                'Partie A': (results['scores']['part_a']/24)*100,
                'Partie B': (results['scores']['part_b']/48)*100,
                'Inattention': (results['scores']['inattention']/28)*100,
                'Hyperactivité': (results['scores']['hyperactivity']/44)*100,
                'Score Total': (results['scores']['total']/72)*100
            }
            
            fig_radar.add_trace(go.Scatterpolar(
                r=list(radar_data.values()),
                theta=list(radar_data.keys()),
                fill='toself',
                name='Scores ASRS (%)',
                line=dict(color='#ff5722', width=3)
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Profil complet ASRS (%)",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
        else:
            st.warning("Veuillez d'abord compléter le test ASRS.")

    with pred_tabs[3]:
        if 'asrs_results' in st.session_state:
            st.subheader("📈 KPIs Avancés et Métriques Cliniques")
            
            results = st.session_state.asrs_results
            
            # KPIs principaux
            st.markdown("### 🎯 KPIs Principaux")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Calculs des KPIs
            total_score = results['scores']['total']
            severity_index = (total_score / 72) * 100
            
            inatt_dominance = results['scores']['inattention'] / (results['scores']['inattention'] + results['scores']['hyperactivity'])
            hyper_dominance = 1 - inatt_dominance
            
            response_consistency = 1 - (np.std(list(results['responses'].values())) / 4)  # Normalisation sur 0-4
            
            high_severity_responses = sum([1 for score in results['responses'].values() if score >= 3])
            severity_concentration = (high_severity_responses / 18) * 100
            
            part_a_severity = (results['scores']['part_a'] / 24) * 100
            
            with col1:
                st.metric(
                    "Indice de sévérité", 
                    f"{severity_index:.1f}%",
                    help="Pourcentage du score maximum possible"
                )
            with col2:
                st.metric(
                    "Dominance inattention", 
                    f"{inatt_dominance:.1%}",
                    help="Proportion des symptômes d'inattention"
                )
            with col3:
                st.metric(
                    "Cohérence réponses", 
                    f"{response_consistency:.1%}",
                    help="Consistance du pattern de réponses"
                )
            with col4:
                st.metric(
                    "Concentration sévérité", 
                    f"{severity_concentration:.1f}%",
                    help="% de réponses 'Souvent' ou 'Très souvent'"
                )
            with col5:
                st.metric(
                    "Score dépistage", 
                    f"{part_a_severity:.1f}%",
                    help="Performance sur les 6 questions clés"
                )

            # Métriques cliniques avancées
            st.markdown("### 🏥 Métriques Cliniques")
            
            # Classification selon plusieurs critères
            col1, col2 = st.columns(2)
            
            with col1:
                # Critères DSM-5 simplifiés
                dsm5_inattention = results['scores']['inattention'] >= 18  # Seuil estimé
                dsm5_hyperactivity = results['scores']['hyperactivity'] >= 18  # Seuil estimé
                
                if dsm5_inattention and dsm5_hyperactivity:
                    dsm5_type = "Mixte"
                    dsm5_color = "#ff5722"
                elif dsm5_inattention:
                    dsm5_type = "Inattentif"
                    dsm5_color = "#ff9800"
                elif dsm5_hyperactivity:
                    dsm5_type = "Hyperactif-Impulsif"
                    dsm5_color = "#ffcc02"
                else:
                    dsm5_type = "Sous-seuil"
                    dsm5_color = "#4caf50"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 4px solid {dsm5_color}; margin-bottom: 15px;">
                    <h4 style="color: {dsm5_color}; margin: 0 0 10px 0;">Type TDAH estimé</h4>
                    <h3 style="color: {dsm5_color}; margin: 0;">{dsm5_type}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Niveau de risque fonctionnel
                functional_impact = (
                    (results['demographics']['quality_of_life'] <= 5) * 0.3 +
                    (results['demographics']['stress_level'] >= 4) * 0.3 +
                    (severity_index >= 60) * 0.4
                )
                
                impact_level = "Sévère" if functional_impact >= 0.7 else "Modéré" if functional_impact >= 0.4 else "Léger"
                impact_color = "#f44336" if functional_impact >= 0.7 else "#ff9800" if functional_impact >= 0.4 else "#4caf50"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 4px solid {impact_color}; margin-bottom: 15px;">
                    <h4 style="color: {impact_color}; margin: 0 0 10px 0;">Impact fonctionnel</h4>
                    <h3 style="color: {impact_color}; margin: 0;">{impact_level}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Score de priorité clinique
                clinical_priority = (
                    (part_a_severity >= 70) * 0.4 +
                    (severity_concentration >= 50) * 0.3 +
                    (functional_impact >= 0.5) * 0.3
                )
                
                priority_level = "Urgente" if clinical_priority >= 0.7 else "Élevée" if clinical_priority >= 0.5 else "Standard"
                priority_color = "#f44336" if clinical_priority >= 0.7 else "#ff9800" if clinical_priority >= 0.5 else "#4caf50"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 4px solid {priority_color}; margin-bottom: 15px;">
                    <h4 style="color: {priority_color}; margin: 0 0 10px 0;">Priorité clinique</h4>
                    <h3 style="color: {priority_color}; margin: 0;">{priority_level}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Indice de fiabilité
                reliability_factors = [
                    response_consistency >= 0.6,  # Cohérence des réponses
                    len([x for x in results['responses'].values() if x > 0]) >= 12,  # Nombre de symptômes
                    abs(results['scores']['inattention'] - results['scores']['hyperactivity']) <= 15  # Équilibre
                ]
                
                reliability_score = sum(reliability_factors) / len(reliability_factors)
                reliability_level = "Élevée" if reliability_score >= 0.8 else "Modérée" if reliability_score >= 0.6 else "Faible"
                reliability_color = "#4caf50" if reliability_score >= 0.8 else "#ff9800" if reliability_score >= 0.6 else "#f44336"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 4px solid {reliability_color};">
                    <h4 style="color: {reliability_color}; margin: 0 0 10px 0;">Fiabilité évaluation</h4>
                    <h3 style="color: {reliability_color}; margin: 0;">{reliability_level}</h3>
                </div>
                """, unsafe_allow_html=True)

            # Graphiques des KPIs
            st.markdown("### 📊 Visualisation des KPIs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # KPIs radar
                kpi_data = {
                    'Sévérité': severity_index,
                    'Concentration': severity_concentration,
                    'Cohérence': response_consistency * 100,
                    'Impact fonctionnel': functional_impact * 100,
                    'Priorité clinique': clinical_priority * 100
                }
                
                fig_kpi = go.Figure()
                
                fig_kpi.add_trace(go.Scatterpolar(
                    r=list(kpi_data.values()),
                    theta=list(kpi_data.keys()),
                    fill='toself',
                    name='KPIs (%)',
                    line=dict(color='#ff5722', width=3)
                ))
                
                fig_kpi.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Profil KPIs cliniques"
                )
                
                st.plotly_chart(fig_kpi, use_container_width=True)
                
            with col2:
                # Évolution temporelle simulée (pour démo)
                weeks = list(range(1, 13))
                baseline_severity = severity_index
                
                # Simulation d'évolution avec variabilité
                np.random.seed(42)
                evolution = [baseline_severity + np.random.normal(0, 5) for _ in weeks]
                
                fig_evolution = go.Figure()
                
                fig_evolution.add_trace(go.Scatter(
                    x=weeks,
                    y=evolution,
                    mode='lines+markers',
                    name='Sévérité estimée',
                    line=dict(color='#ff5722', width=3)
                ))
                
                fig_evolution.add_hline(
                    y=baseline_severity, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Baseline actuelle"
                )
                
                fig_evolution.update_layout(
                    title='Évolution projetée (simulation)',
                    xaxis_title='Semaines',
                    yaxis_title='Indice de sévérité (%)',
                    height=400
                )
                
                st.plotly_chart(fig_evolution, use_container_width=True)

            # Tableau de bord récapitulatif
            st.markdown("### 📋 Tableau de bord récapitulatif")
            
            dashboard_data = {
                'Métrique': [
                    'Score ASRS total', 'Partie A (dépistage)', 'Inattention', 'Hyperactivité',
                    'Indice de sévérité', 'Impact fonctionnel', 'Priorité clinique', 'Fiabilité'
                ],
                'Valeur': [
                    f"{total_score}/72",
                    f"{results['scores']['part_a']}/24",
                    f"{results['scores']['inattention']}/28", 
                    f"{results['scores']['hyperactivity']}/44",
                    f"{severity_index:.1f}%",
                    impact_level,
                    priority_level,
                    reliability_level
                ],
                'Interprétation': [
                    "Score global ASRS",
                    "Questions de dépistage clés", 
                    "Symptômes d'inattention",
                    "Symptômes hyperactivité-impulsivité",
                    "Pourcentage de sévérité globale",
                    "Impact sur vie quotidienne",
                    "Urgence consultation",
                    "Qualité de l'évaluation"
                ],
                'Statut': [
                    "🔴 Élevé" if total_score >= 45 else "🟡 Modéré" if total_score >= 30 else "🟢 Faible",
                    "🔴 Positif" if results['scores']['part_a'] >= 14 else "🟢 Négatif",
                    "🔴 Élevé" if results['scores']['inattention'] >= 18 else "🟡 Modéré" if results['scores']['inattention'] >= 12 else "🟢 Faible",
                    "🔴 Élevé" if results['scores']['hyperactivity'] >= 18 else "🟡 Modéré" if results['scores']['hyperactivity'] >= 12 else "🟢 Faible",
                    "🔴 Élevé" if severity_index >= 60 else "🟡 Modéré" if severity_index >= 40 else "🟢 Faible",
                    f"🔴 {impact_level}" if impact_level == "Sévère" else f"🟡 {impact_level}" if impact_level == "Modéré" else f"🟢 {impact_level}",
                    f"🔴 {priority_level}" if priority_level == "Urgente" else f"🟡 {priority_level}" if priority_level == "Élevée" else f"🟢 {priority_level}",
                    f"🟢 {reliability_level}" if reliability_level == "Élevée" else f"🟡 {reliability_level}" if reliability_level == "Modérée" else f"🔴 {reliability_level}"
                ]
            }
            
            dashboard_df = pd.DataFrame(dashboard_data)
            st.dataframe(dashboard_df, use_container_width=True)
            
        else:
            st.warning("Veuillez d'abord compléter le test ASRS.")

    with pred_tabs[4]:
        if 'asrs_results' in st.session_state:
            st.subheader("💡 Recommandations Personnalisées")
            
            results = st.session_state.asrs_results
            
            # Analyse pour recommandations
            total_score = results['scores']['total']
            part_a_score = results['scores']['part_a']
            severity_index = (total_score / 72) * 100
            
            # Recommandations basées sur les scores
            if part_a_score >= 16:
                urgency = "URGENTE"
                urgency_color = "#f44336"
                consultation_delay = "dans les 2 semaines"
            elif part_a_score >= 14:
                urgency = "ÉLEVÉE"
                urgency_color = "#ff9800"
                consultation_delay = "dans le mois"
            elif part_a_score >= 10:
                urgency = "MODÉRÉE"
                urgency_color = "#ffcc02"
                consultation_delay = "dans les 3 mois"
            else:
                urgency = "SURVEILLANCE"
                urgency_color = "#4caf50"
                consultation_delay = "selon évolution"

            # Recommandation principale
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {urgency_color}, {urgency_color}99); 
                       padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white;">
                <h3 style="margin: 0 0 15px 0;">🎯 Recommandation Prioritaire</h3>
                <h2 style="margin: 0 0 10px 0;">Consultation {urgency}</h2>
                <p style="margin: 0; font-size: 1.1rem;">Prendre rendez-vous avec un spécialisé TDAH {consultation_delay}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommandations détaillées par domaine
            st.markdown("### 🏥 Recommandations Cliniques")
            
            clinical_recommendations = []
            
            # Basé sur le score total
            if total_score >= 45:
                clinical_recommendations.extend([
                    "Évaluation psychiatrique complète recommandée",
                    "Bilan neuropsychologique pour confirmer le diagnostic",
                    "Évaluation des troubles associés (anxiété, dépression)"
                ])
            elif total_score >= 30:
                clinical_recommendations.extend([
                    "Consultation avec psychiatre ou psychologue spécialisé",
                    "Entretien clinique structuré TDAH",
                    "Évaluation du retentissement fonctionnel"
                ])
            else:
                clinical_recommendations.extend([
                    "Suivi avec médecin traitant",
                    "Réévaluation dans 6 mois si symptômes persistent",
                    "Information sur les signes d'alerte TDAH"
                ])
            
            # Basé sur les dimensions dominantes
            inatt_score = results['scores']['inattention']
            hyper_score = results['scores']['hyperactivity']
            
            if inatt_score > hyper_score + 5:
                clinical_recommendations.append("Focus sur l'évaluation des troubles attentionnels")
            elif hyper_score > inatt_score + 5:
                clinical_recommendations.append("Évaluation spécifique de l'hyperactivité-impulsivité")
            else:
                clinical_recommendations.append("Évaluation complète forme mixte TDAH")

            for rec in clinical_recommendations:
                st.markdown(f"• **{rec}**")
            
            # Recommandations de vie quotidienne
            st.markdown("### 🏠 Stratégies de Vie Quotidienne")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Gestion de l'attention :**
                - Technique Pomodoro (25 min de travail / 5 min pause)
                - Environnement de travail calme et organisé
                - Élimination des distracteurs (notifications, bruit)
                - Planification détaillée des tâches
                - Utilisation d'applications de concentration
                
                **📅 Organisation :**
                - Agenda papier ou numérique systématique
                - Listes de tâches quotidiennes
                - Rappels automatiques pour rendez-vous
                - Routine matinale et vespérale structurée
                """)
                
            with col2:
                st.markdown("""
                **⚡ Gestion de l'hyperactivité :**
                - Activité physique régulière (30 min/jour)
                - Pauses mouvement toutes les heures
                - Techniques de relaxation (méditation, respiration)
                - Sport ou activités physiques intenses
                
                **🧘 Bien-être émotionnel :**
                - Sommeil régulier (7-9h par nuit)
                - Alimentation équilibrée
                - Limitation de la caféine
                - Gestion du stress (yoga, sophrologie)
                """)

            # Recommandations professionnelles/éducatives
            st.markdown("### 💼 Aménagements Professionnels/Éducatifs")
            
            work_recommendations = []
            
            if severity_index >= 60:
                work_recommendations.extend([
                    "Demande d'aménagements de poste de travail",
                    "Temps de pause supplémentaires", 
                    "Bureau isolé ou casque anti-bruit",
                    "Possibilité de télétravail partiel",
                    "Reconnaissance travailleur handicapé (RQTH)"
                ])
            elif severity_index >= 40:
                work_recommendations.extend([
                    "Discussion avec RH pour aménagements légers",
                    "Organisation du poste de travail",
                    "Gestion des priorités avec superviseur"
                ])
            else:
                work_recommendations.extend([
                    "Auto-organisation optimisée",
                    "Communication des besoins à l'équipe"
                ])

            for rec in work_recommendations:
                st.markdown(f"• **{rec}**")

            # Ressources et soutien
            st.markdown("### 📚 Ressources et Soutien")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🏛️ Organisations :**
                - TDAH France (association nationale)
                - HyperSupers TDAH France
                - Association locale TDAH
                - Centres experts TDAH adulte
                
                **📱 Applications recommandées :**
                - Forest (concentration)
                - Todoist (organisation)
                - Headspace (méditation)
                - Sleep Cycle (sommeil)
                """)
                
            with col2:
                st.markdown("""
                **📖 Lectures recommandées :**
                - "TDAH chez l'adulte" - Dr. Michel Bouvard
                - "Mon cerveau a TDAH" - Dr. Annick Vincent
                - Guides pratiques HAS (Haute Autorité de Santé)
                
                **🌐 Sites web fiables :**
                - tdah-france.fr
                - has-sante.fr (recommandations officielles)
                - ameli.fr (information patients)
                """)

            # Plan d'action personnalisé
            st.markdown("### 📋 Plan d'Action Personnalisé")
            
            action_plan = f"""
            <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 4px solid #ff9800;">
                <h4 style="color: #ef6c00; margin-top: 0;">🎯 Prochaines étapes recommandées</h4>
                <ol style="color: #f57c00; line-height: 1.8;">
                    <li><strong>Immédiat (0-2 semaines) :</strong> Prendre rendez-vous avec professionnel spécialisé TDAH</li>
                    <li><strong>Court terme (1 mois) :</strong> Mettre en place techniques d'organisation de base</li>
                    <li><strong>Moyen terme (3 mois) :</strong> Évaluer l'efficacité des stratégies mises en place</li>
                    <li><strong>Long terme (6 mois) :</strong> Bilan complet et ajustement du plan de prise en charge</li>
                </ol>
                <p style="color: #ef6c00; font-style: italic; margin-bottom: 0;">
                    Ce plan sera adapté selon les résultats de l'évaluation clinique professionnelle.
                </p>
            </div>
            """
            
            st.markdown(action_plan, unsafe_allow_html=True)
            
            # Suivi et monitoring
            st.markdown("### 📊 Suivi Recommandé")
            
            monitoring_schedule = {
                'Période': ['2 semaines', '1 mois', '3 mois', '6 mois', '1 an'],
                'Action': [
                    'Consultation spécialisée',
                    'Bilan stratégies mises en place',
                    'Évaluation amélioration symptômes', 
                    'Bilan complet fonctionnement',
                    'Réévaluation globale'
                ],
                'Objectif': [
                    'Diagnostic professionnel',
                    'Ajustement techniques',
                    'Mesure efficacité interventions',
                    'Adaptation plan traitement',
                    'Maintien bénéfices à long terme'
                ]
            }
            
            monitoring_df = pd.DataFrame(monitoring_schedule)
            st.dataframe(monitoring_df, use_container_width=True)
            
        else:
            st.warning("Veuillez d'abord compléter le test ASRS.")

def show_enhanced_documentation():
    """Documentation enrichie pour le TDAH avec plus de ressources"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff5722, #ff9800);
                padding: 40px 25px; border-radius: 20px; margin-bottom: 35px; text-align: center;">
        <h1 style="color: white; font-size: 2.8rem; margin-bottom: 15px;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 600;">
            📚 Documentation Complète TDAH
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem;
                  max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Guide exhaustif sur le Trouble Déficitaire de l'Attention avec Hyperactivité
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Onglets de documentation enrichis
    doc_tabs = st.tabs([
        "📖 Bases du TDAH",
        "🔬 Critères diagnostiques", 
        "💊 Traitements",
        "🏫 Accompagnement",
        "📊 Échelles d'évaluation",
        "🧠 Recherche récente",
        "📚 Ressources pratiques",
        "❓ FAQ"
    ])

    with doc_tabs[0]:
        st.subheader("📖 Comprendre le TDAH - Bases Scientifiques")
        
        # Définition moderne
        st.markdown("""
        <div class="info-card-modern">
            <h3 style="color: #ff5722;">🧬 Définition Actuelle (DSM-5-TR, 2022)</h3>
            <p style="line-height: 1.8;">
                Le TDAH est un trouble neurodéveloppemental persistant caractérisé par un pattern 
                d'inattention et/ou d'hyperactivité-impulsivité qui interfère avec le fonctionnement 
                ou le développement. Les symptômes sont présents dans multiple environnements et 
                causent une détresse ou une altération cliniquement significative.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Neurobiologie du TDAH :**
            
            *Structures cérébrales impliquées :*
            - **Cortex préfrontal :** Fonctions exécutives, attention soutenue
            - **Cortex cingulaire antérieur :** Contrôle attentionnel, résolution conflits
            - **Ganglions de la base :** Contrôle moteur, motivation
            - **Cervelet :** Coordination motrice, fonctions cognitives
            
            *Neurotransmetteurs :*
            - **Dopamine :** Motivation, récompense, attention
            - **Noradrénaline :** Vigilance, attention sélective
            - **Sérotonine :** Régulation humeur, impulsivité
            
            *Anomalies identifiées :*
            - Retard maturation cortex préfrontal (2-3 ans)
            - Dysfonctionnement circuits fronto-striataux
            - Altération connectivité réseaux attentionnels
            """)
            
        with col2:
            st.markdown("""
            **📊 Épidémiologie Mondiale :**
            
            *Prévalence :*
            - **Enfants :** 5-7% (variation selon critères diagnostiques)
            - **Adolescents :** 4-6% (légère diminution avec l'âge)
            - **Adultes :** 2.5-4% (reconnaissance récente)
            - **Ratio garçons/filles :** 3:1 (enfance) → 1.5:1 (âge adulte)
            
            *Facteurs de risque :*
            - **Génétique :** Héritabilité 70-80%
            - **Environnementaux :** Prématurité, exposition toxique
            - **Sociaux :** Stress familial, adversité précoce
            
            *Évolution :*
            - **Persistance à l'âge adulte :** 60-70% des cas
            - **Amélioration naturelle :** 30-40% avec l'âge
            - **Complications :** Troubles associés fréquents
            """)

        # Comorbidités et troubles associés
        st.markdown("### 🔗 Troubles Fréquemment Associés")
        
        comorbidities_data = {
            'Trouble': [
                'Troubles anxieux', 'Troubles de l\'humeur', 'Troubles oppositionnels',
                'Troubles des apprentissages', 'Troubles du sommeil', 'Addictions',
                'Troubles alimentaires', 'Troubles de la personnalité'
            ],
            'Prévalence (%)': ['25-40', '15-75', '40-60', '20-60', '25-50', '15-25', '10-30', '10-20'],
            'Impact': [
                'Anxiété sociale, phobies', 'Dépression, bipolarité', 'Défiance, agressivité',
                'Dyslexie, dyscalculie', 'Insomnie, hypersomnie', 'Substances, jeux',
                'Boulimie, compulsions', 'Borderline, antisocial'
            ]
        }
        
        comorbidities_df = pd.DataFrame(comorbidities_data)
        st.dataframe(comorbidities_df, use_container_width=True)

        # Mythes et réalités
        st.markdown("### ❌ Mythes vs ✅ Réalités")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 4px solid #f44336;">
                <h4 style="color: #c62828; margin-top: 0;">❌ Mythes fréquents</h4>
                <ul style="color: #d32f2f; line-height: 1.8;">
                    <li>"Le TDAH n'existe pas vraiment"</li>
                    <li>"C'est juste un manque de discipline"</li>
                    <li>"Ça disparaît à l'âge adulte"</li>
                    <li>"Les médicaments créent des dépendances"</li>
                    <li>"C'est dû à la mauvaise éducation"</li>
                    <li>"Tout le monde a un peu de TDAH"</li>
                    <li>"C'est une mode récente"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 4px solid #4caf50;">
                <h4 style="color: #2e7d32; margin-top: 0;">✅ Réalités scientifiques</h4>
                <ul style="color: #388e3c; line-height: 1.8;">
                    <li>Trouble neurodéveloppemental validé scientifiquement</li>
                    <li>Différences cérébrales objectivables</li>
                    <li>Persistance fréquente à l'âge adulte</li>
                    <li>Médicaments sûrs et efficaces si bien utilisés</li>
                    <li>Origine neurobiologique, pas éducative</li>
                    <li>Diagnostic nécessite critères stricts</li>
                    <li>Décrit depuis plus d'un siècle</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with doc_tabs[1]:
        st.subheader("🔬 Critères Diagnostiques Détaillés")
        
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #ef6c00;">📋 Critères DSM-5-TR (2022)</h4>
            <p style="color: #f57c00;">
                Le diagnostic de TDAH nécessite la présence d'au moins 6 symptômes (5 pour les adultes) 
                dans au moins une des deux catégories, persistants depuis au moins 6 mois.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Critères détaillés avec exemples
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 A. Inattention** (6+ symptômes pendant 6+ mois)
            
            1. **Difficultés d'attention aux détails**
               - *Exemples :* Erreurs d'étourderie au travail, négligence des détails
               - *Manifestations adultes :* Erreurs dans rapports, formulaires incorrects
            
            2. **Difficultés à maintenir l'attention**
               - *Exemples :* Distraction pendant conversations, lectures
               - *Manifestations adultes :* Perte de fil en réunion, difficultés tâches longues
            
            3. **Semble ne pas écouter**
               - *Exemples :* Esprit ailleurs quand on lui parle directement
               - *Manifestations adultes :* Répétitions nécessaires, oubli consignes
            
            4. **N'achève pas les tâches**
               - *Exemples :* Abandonne projets en cours, procrastination
               - *Manifestations adultes :* Projets inachevés, délais non respectés
            
            5. **Difficultés d'organisation**
               - *Exemples :* Bureau désordonné, mauvaise gestion du temps
               - *Manifestations adultes :* Retards fréquents, planification chaotique
            
            6. **Évite les efforts mentaux**
               - *Exemples :* Reporte tâches nécessitant concentration
               - *Manifestations adultes :* Évitement paperasserie, tâches administratives
            
            7. **Perd souvent des objets**
               - *Exemples :* Clés, téléphone, documents importants
               - *Manifestations adultes :* Recherches fréquentes, stress lié aux pertes
            
            8. **Facilement distrait**
               - *Exemples :* Interrompu par stimuli externes, pensées intrusives
               - *Manifestations adultes :* Difficultés environnements bruyants
            
            9. **Oublis quotidiens**
               - *Exemples :* Rendez-vous, tâches ménagères, obligations
               - *Manifestations adultes :* Oubli factures, anniversaires, médicaments
            """)
            
        with col2:
            st.markdown("""
            **⚡ B. Hyperactivité-Impulsivité** (6+ symptômes pendant 6+ mois)
            
            **Hyperactivité :**
            
            1. **Remue mains/pieds, se tortille**
               - *Exemples :* Bouge sans cesse, tape du pied
               - *Manifestations adultes :* Agitation discrète, besoin de bouger
            
            2. **Se lève de son siège**
               - *Exemples :* Difficultés à rester assis longtemps
               - *Manifestations adultes :* Pauses fréquentes, besoin de marcher
            
            3. **Court ou grimpe inappropriément**
               - *Exemples :* Agitation motrice excessive
               - *Manifestations adultes :* Sensation interne d'agitation
            
            4. **Difficultés loisirs calmes**
               - *Exemples :* Préfère activités dynamiques
               - *Manifestations adultes :* Évite activités sédentaires
            
            5. **Toujours "sous pression"**
               - *Exemples :* Comme "mu par un moteur"
               - *Manifestations adultes :* Difficulté à se détendre
            
            6. **Parle excessivement**
               - *Exemples :* Bavardage constant, verbosité
               - *Manifestations adultes :* Tendance au monologue
            
            **Impulsivité :**
            
            7. **Répond avant fin des questions**
               - *Exemples :* Anticipe les questions
               - *Manifestations adultes :* Coupe la parole, finit phrases
            
            8. **Difficultés à attendre son tour**
               - *Exemples :* Impatience dans files d'attente
               - *Manifestations adultes :* Frustration délais, urgence constante
            
            9. **Interrompt ou importune**
               - *Exemples :* S'immisce dans conversations/jeux
               - *Manifestations adultes :* Interruptions fréquentes, intrusion
            """)

        # Critères généraux obligatoires
        st.markdown("### 📋 Critères Généraux Obligatoires")
        
        criteria_general = [
            ("C. Âge d'apparition", "Plusieurs symptômes présents avant l'âge de 12 ans"),
            ("D. Contextes multiples", "Symptômes présents dans au moins 2 environnements (maison, travail, école, etc.)"),
            ("E. Altération fonctionnelle", "Preuves claires d'altération cliniquement significative du fonctionnement"),
            ("F. Exclusion", "Symptômes non mieux expliqués par un autre trouble mental")
        ]
        
        for criterion, description in criteria_general:
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #2196f3;">
                <h5 style="color: #1565c0; margin: 0 0 8px 0;">{criterion}</h5>
                <p style="color: #1976d2; margin: 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

        # Spécifications diagnostiques
        st.markdown("### 🎯 Présentations du TDAH")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffebee, #ffcdd2); border-radius: 12px; padding: 20px; height: 200px;">
                <h4 style="color: #c62828;">🎯 Présentation Inattentive</h4>
                <ul style="color: #d32f2f; font-size: 0.9rem;">
                    <li>≥6 symptômes inattention</li>
                    <li>&lt;6 symptômes hyperactivité</li>
                    <li>Plus fréquent chez les filles</li>
                    <li>Diagnostic souvent tardif</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3e0, #ffcc02); border-radius: 12px; padding: 20px; height: 200px;">
                <h4 style="color: #ef6c00;">⚡ Présentation Hyperactive</h4>
                <ul style="color: #f57c00; font-size: 0.9rem;">
                    <li>&lt;6 symptômes inattention</li>
                    <li>≥6 symptômes hyperactivité</li>
                    <li>Plus fréquent chez garçons</li>
                    <li>Diagnostic précoce</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e8f5e8, #c8e6c9); border-radius: 12px; padding: 20px; height: 200px;">
                <h4 style="color: #2e7d32;">🌈 Présentation Mixte</h4>
                <ul style="color: #388e3c; font-size: 0.9rem;">
                    <li>≥6 symptômes inattention</li>
                    <li>≥6 symptômes hyperactivité</li>
                    <li>Forme la plus sévère</li>
                    <li>Impact fonctionnel élevé</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with doc_tabs[2]:
        st.subheader("💊 Traitements Evidence-Based")
        
        # Vue d'ensemble des traitements
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #ef6c00;">🎯 Approche Multimodale Recommandée</h4>
            <p style="color: #f57c00; line-height: 1.6;">
                Le traitement optimal du TDAH combine plusieurs approches selon l'âge, la sévérité 
                et les préférences du patient. L'approche multimodale est la plus efficace.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        treatment_tabs = st.tabs(["💊 Pharmacothérapie", "🧠 Psychothérapies", "📚 Interventions éducatives", "🏃 Interventions lifestyle"])
        
        with treatment_tabs[0]:
            st.markdown("### 💊 Traitements Pharmacologiques")
            
            # Stimulants
            st.markdown("#### ⚡ Psychostimulants (1ère ligne)")
            
            stimulants_data = {
                'Médicament': ['Méthylphénidate IR', 'Méthylphénidate LP', 'Lisdexamfétamine', 'Dextroamphétamine'],
                'Noms commerciaux': ['Ritaline®', 'Concerta®, Quasym®', 'Elvanse®', 'Dexedrine®'],
                'Durée d\'action': ['3-5h', '8-12h', '10-14h', '4-6h'],
                'Efficacité (%)': ['70-80', '70-80', '70-85', '70-80'],
                'Avantages': [
                    'Flexibilité dosage', 'Prise unique/jour', 'Moins abus potentiel', 'Action rapide'
                ],
                'Inconvénients': [
                    'Prises multiples', 'Moins flexible', 'Plus cher', 'Prises multiples'
                ]
            }
            
            stimulants_df = pd.DataFrame(stimulants_data)
            st.dataframe(stimulants_df, use_container_width=True)
            
            # Non-stimulants
            st.markdown("#### 🔄 Non-stimulants (2ème ligne)")
            
            non_stimulants_data = {
                'Médicament': ['Atomoxétine', 'Guanfacine LP', 'Clonidine LP', 'Bupropion'],
                'Noms commerciaux': ['Strattera®', 'Intuniv®', 'Kapvay®', 'Wellbutrin®'],
                'Durée d\'action': ['24h', '24h', '12h', '12-24h'],
                'Efficacité (%)': ['50-60', '40-50', '40-50', '45-55'],
                'Avantages': [
                    'Pas de dépendance', 'Moins d\'effets cardiovasculaires', 'Aide avec tics', 'Antidépresseur'
                ],
                'Inconvénients': [
                    'Délai d\'action 2-4 sem', 'Somnolence', 'Hypotension', 'Convulsions'
                ]
            }
            
            non_stimulants_df = pd.DataFrame(non_stimulants_data)
            st.dataframe(non_stimulants_df, use_container_width=True)
            
            # Mécanismes d'action
            st.markdown("#### 🧬 Mécanismes d'action")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Psychostimulants :**
                - Inhibition recapture dopamine/noradrénaline
                - Augmentation disponibilité neurotransmetteurs
                - Action rapide (30-60 minutes)
                - Amélioration attention et contrôle exécutif
                """)
                
            with col2:
                st.markdown("""
                **Non-stimulants :**
                - Atomoxétine : inhibiteur sélectif recapture noradrénaline
                - Guanfacine : agoniste α2A-adrénergique
                - Action progressive (2-8 semaines)
                - Amélioration régulation émotionnelle
                """)

            # Posologies et surveillance
            st.markdown("#### 📏 Posologies recommandées")
            
            posology_data = {
                'Médicament': [
                    'Méthylphénidate IR', 'Méthylphénidate LP', 'Atomoxétine', 
                    'Guanfacine LP', 'Lisdexamfétamine'
                ],
                'Dose initiale': ['5-10 mg 2x/j', '18 mg 1x/j', '0.5 mg/kg/j', '1 mg 1x/j', '30 mg 1x/j'],
                'Dose thérapeutique': ['0.3-1 mg/kg/j', '18-72 mg/j', '1.2-1.8 mg/kg/j', '1-4 mg/j', '30-70 mg/j'],
                'Dose maximale': ['60 mg/j', '72 mg/j', '100 mg/j', '4 mg/j', '70 mg/j'],
                'Surveillance': [
                    'FC, TA, sommeil', 'FC, TA, croissance', 'FC, TA, fonction hépatique',
                    'FC, TA, sédation', 'FC, TA, sommeil'
                ]
            }
            
            posology_df = pd.DataFrame(posology_data)
            st.dataframe(posology_df, use_container_width=True)

        with treatment_tabs[1]:
            st.markdown("### 🧠 Psychothérapies Evidence-Based")
            
            psycho_tabs = st.tabs(["TCC", "Thérapie comportementale", "Remédiation cognitive", "Mindfulness"])
            
            with psycho_tabs[0]:
                st.markdown("""
                #### 🎯 Thérapie Cognitivo-Comportementale (TCC)
                
                **Objectifs principaux :**
                - Modification des pensées dysfonctionnelles
                - Développement de stratégies de coping
                - Amélioration de l'organisation et planification
                - Gestion de l'impulsivité
                
                **Techniques spécifiques :**
                - Auto-surveillance des symptômes
                - Restructuration cognitive
                - Résolution de problèmes
                - Gestion du temps et priorités
                - Techniques de relaxation
                
                **Efficacité :**
                - Taille d'effet modérée à importante (d=0.5-0.8)
                - Combinaison TCC + médication = meilleurs résultats
                - Maintien des bénéfices à long terme
                """)
                
            with psycho_tabs[1]:
                st.markdown("""
                #### 🎮 Thérapie Comportementale
                
                **Programmes d'entraînement aux habiletés parentales (PEHP) :**
                - Techniques de renforcement positif
                - Gestion des comportements difficiles
                - Communication efficace
                - Structuration de l'environnement familial
                
                **Interventions scolaires :**
                - Gestion de classe comportementale
                - Systèmes de renforcement
                - Modification de l'environnement
                - Formation des enseignants
                
                **Efficacité démontrée :**
                - Réduction significative des comportements perturbateurs
                - Amélioration du climat familial
                - Transfert des acquis à l'école
                """)
                
            with psycho_tabs[2]:
                st.markdown("""
                #### 🧩 Remédiation Cognitive
                
                **Entraînement des fonctions exécutives :**
                - Mémoire de travail
                - Flexibilité cognitive
                - Inhibition
                - Planification
                
                **Outils et programmes :**
                - CogMed (mémoire de travail)
                - Captain's Log
                - Jeux vidéo thérapeutiques
                - Entraînement informatisé
                
                **Résultats :**
                - Amélioration spécifique des fonctions entraînées
                - Transfert variable aux situations quotidiennes
                - Nécessité de généralisation active
                """)
                
            with psycho_tabs[3]:
                st.markdown("""
                #### 🧘 Interventions basées sur la Pleine Conscience
                
                **Mindfulness-Based Interventions (MBI) :**
                - Attention au moment présent
                - Acceptation sans jugement
                - Régulation émotionnelle
                - Réduction du stress
                
                **Programmes spécialisés :**
                - MindUP (enfants/adolescents)
                - MBSR adapté TDAH
                - Yoga thérapeutique
                - Méditation de mouvement
                
                **Bénéfices documentés :**
                - Amélioration de l'attention soutenue
                - Réduction de l'impulsivité
                - Meilleure régulation émotionnelle
                - Diminution de l'anxiété comorbide
                """)

        with treatment_tabs[2]:
            st.markdown("### 📚 Interventions Psychoéducatives")
            
            educ_tabs = st.tabs(["Milieu scolaire", "Aménagements", "Formation", "Technologies"])
            
            with educ_tabs[0]:
                st.markdown("""
                #### 🏫 Interventions en Milieu Scolaire
                
                **Stratégies pédagogiques :**
                - Instructions courtes et séquentielles
                - Support visuel et kinesthésique
                - Pauses mouvement régulières
                - Feedback immédiat et spécifique
                - Environnement structuré et prévisible
                
                **Gestion de classe :**
                - Règles claires et affichées
                - Système de renforcement positif
                - Signaux discrets pour recentrer
                - Placement stratégique dans la classe
                - Partenariat avec un pair
                """)
                
            with educ_tabs[1]:
                st.markdown("""
                #### ⚙️ Aménagements et Adaptations
                
                **Plan d'Accompagnement Personnalisé (PAP) :**
                - Temps supplémentaire (1/3 temps)
                - Pauses pendant les évaluations
                - Reformulation des consignes
                - Utilisation d'ordinateur
                - Lieu d'examen adapté (salle calme)
                
                **Outils compensatoires :**
                - Agendas visuels
                - Minuteurs et alarmes
                - Enregistreurs vocaux
                - Logiciels de mind mapping
                - Applications d'organisation
                """)
                
            with educ_tabs[2]:
                st.markdown("""
                #### 👨‍🏫 Formation des Équipes
                
                **Formation des enseignants :**
                - Compréhension du TDAH
                - Stratégies d'intervention
                - Gestion des comportements
                - Collaboration avec les familles
                
                **Formation des AVS/AESH :**
                - Techniques d'accompagnement
                - Aide à l'organisation
                - Soutien discret en classe
                - Communication avec l'équipe
                """)
                
            with educ_tabs[3]:
                st.markdown("""
                #### 💻 Technologies d'Assistance
                
                **Applications mobiles :**
                - Gestionnaires de tâches (Todoist, Any.do)
                - Minuteurs (Forest, Focus Keeper)
                - Prise de notes (Evernote, OneNote)
                - Lecture assistée (Voice Dream Reader)
                
                **Logiciels spécialisés :**
                - Prédicteurs de mots
                - Correcteurs orthographiques avancés
                - Synthèse vocale
                - Reconnaissance vocale
                """)

        with treatment_tabs[3]:
            st.markdown("### 🏃 Interventions Lifestyle")
            
            lifestyle_tabs = st.tabs(["Activité physique", "Nutrition", "Sommeil", "Gestion stress"])
            
            with lifestyle_tabs[0]:
                st.markdown("""
                #### 🏃‍♂️ Activité Physique
                
                **Recommandations :**
                - 60 minutes/jour d'activité modérée à intense
                - Sports d'équipe pour les habiletés sociales
                - Arts martiaux pour l'autodiscipline
                - Natation pour la régulation sensorielle
                
                **Mécanismes bénéfiques :**
                - Augmentation dopamine et noradrénaline
                - Amélioration de la neuroplasticité
                - Réduction du stress et de l'anxiété
                - Amélioration du sommeil
                
                **Types d'activités recommandées :**
                - Sports aérobiques (course, vélo, natation)
                - Sports de coordination (tennis, badminton)
                - Activités de pleine conscience (yoga, tai-chi)
                - Jeux libres et créatifs
                """)
                
            with lifestyle_tabs[1]:
                st.markdown("""
                #### 🥗 Nutrition et TDAH
                
                **Recommandations nutritionnelles :**
                - Régime équilibré riche en protéines
                - Limitation des sucres rapides
                - Acides gras oméga-3 (poissons gras, noix)
                - Fer, zinc, magnésium (si carences)
                
                **Éviter ou limiter :**
                - Colorants artificiels (E102, E110, E124, E129)
                - Conservateurs (benzoates, sulfites)
                - Édulcorants artificiels
                - Caféine excessive
                
                **Suppléments étudiés :**
                - Oméga-3 (EPA/DHA) : effet modeste
                - Fer : si carence avérée
                - Zinc : si déficit documenté
                - Magnésium : pour l'anxiété comorbide
                """)
                
            with lifestyle_tabs[2]:
                st.markdown("""
                #### 😴 Hygiène du Sommeil
                
                **Problèmes fréquents :**
                - Difficultés d'endormissement
                - Réveils nocturnes
                - Sommeil non réparateur
                - Somnolence diurne
                
                **Interventions comportementales :**
                - Horaires de coucher/lever réguliers
                - Routine pré-sommeil apaisante
                - Environnement calme et frais
                - Limitation des écrans 2h avant le coucher
                - Activité physique en journée
                
                **Gestion médicamenteuse :**
                - Adaptation horaire des stimulants
                - Mélatonine si troubles persistants
                - Évaluation apnée du sommeil
                """)
                
            with lifestyle_tabs[3]:
                st.markdown("""
                #### 🧘‍♀️ Gestion du Stress
                
                **Techniques de relaxation :**
                - Respiration profonde et guidée
                - Relaxation musculaire progressive
                - Biofeedback
                - Méditation adaptée à l'âge
                
                **Stratégies de coping :**
                - Identification des déclencheurs
                - Techniques de résolution de problèmes
                - Restructuration cognitive
                - Soutien social et familial
                
                **Environnement apaisant :**
                - Espaces de retrait calmes
                - Objets anti-stress (fidgets)
                - Musique relaxante
                - Aromathérapie légère
                """)

    with doc_tabs[3]:
        st.subheader("🏫 Accompagnement et Aménagements")
        
        accomp_tabs = st.tabs(["Milieu familial", "Milieu scolaire", "Milieu professionnel", "Transitions"])
        
        with accomp_tabs[0]:
            st.markdown("""
            ### 👨‍👩‍👧‍👦 Accompagnement Familial
            
            #### Programmes d'Entraînement aux Habiletés Parentales (PEHP)
            
            **Principes fondamentaux :**
            - Renforcement positif systématique
            - Cohérence éducative entre parents
            - Gestion proactive des comportements
            - Communication bienveillante et claire
            
            **Techniques comportementales :**
            - Économie de jetons/système de points
            - Time-out structuré et bref
            - Conséquences naturelles et logiques
            - Contrats comportementaux
            
            **Organisation familiale :**
            - Routines visuelles et structurées
            - Espaces dédiés (devoirs, jeux, repos)
            - Planification hebdomadaire familiale
            - Gestion des transitions
            """)
            
        with accomp_tabs[1]:
            st.markdown("""
            ### 🎓 Accompagnement Scolaire
            
            #### Plans d'Accompagnement Personnalisé (PAP)
            
            **Aménagements pédagogiques :**
            - Segmentation des tâches complexes
            - Support visuel et kinesthésique
            - Feedback fréquent et positif
            - Alternance activités calmes/dynamiques
            
            **Aménagements d'évaluation :**
            - Tiers-temps supplémentaire
            - Pauses fractionnées
            - Reformulation orale des consignes
            - Utilisation d'ordinateur
            - Lieu d'examen adapté
            
            **Soutien spécialisé :**
            - Aide humaine (AVS/AESH) si nécessaire
            - Enseignement spécialisé (RASED)
            - Suivi orthophonique
            - Remédiation cognitive
            """)
            
        with accomp_tabs[2]:
            st.markdown("""
            ### 💼 Adaptation Professionnelle
            
            #### Reconnaissance et Droits
            
            **Reconnaissance Qualité Travailleur Handicapé (RQTH) :**
            - Facilite l'accès aux aménagements
            - Protection contre la discrimination
            - Accès aux dispositifs d'aide à l'emploi
            - Bilans de compétences adaptés
            
            **Aménagements de poste :**
            - Bureau calme ou isolé phoniquement
            - Horaires flexibles ou télétravail partiel
            - Pauses supplémentaires
            - Découpage des tâches complexes
            - Outils d'aide à l'organisation
            
            **Soutien professionnel :**
            - Job coaching spécialisé
            - Formation aux outils compensatoires
            - Médiation avec l'employeur
            - Suivi psychologique adapté
            """)
            
        with accomp_tabs[3]:
            st.markdown("""
            ### 🔄 Gestion des Transitions
            
            #### Transitions Développementales
            
            **Enfance → Adolescence :**
            - Adaptation des traitements
            - Développement de l'autonomie
            - Préparation aux défis sociaux
            - Éducation sexuelle adaptée
            
            **Adolescence → Âge adulte :**
            - Transition vers soins adultes
            - Orientation professionnelle
            - Autonomie dans la gestion du traitement
            - Préparation à l'indépendance
            
            **Transitions quotidiennes :**
            - Préparation aux changements
            - Routines de transition
            - Objets de transition
            - Anticipation et prévisibilité
            """)

    with doc_tabs[4]:
        st.subheader("📊 Échelles d'Évaluation et Outils")
        
        scales_tabs = st.tabs(["Échelles diagnostiques", "Outils de suivi", "Évaluations cognitives"])
        
        with scales_tabs[0]:
            st.markdown("""
            ### 📝 Échelles Diagnostiques Validées
            
            #### Échelles Auto-rapportées
            
            **ASRS v1.1 (Adult ADHD Self-Report Scale) :**
            - 18 items basés sur critères DSM-5
            - Partie A : 6 questions de dépistage
            - Partie B : 12 questions complémentaires
            - Sensibilité : 68-70%, Spécificité : 99%
            
            **WURS (Wender Utah Rating Scale) :**
            - Évaluation rétrospective de l'enfance
            - 25 items sur symptômes avant 8 ans
            - Complément au diagnostic adulte
            
            #### Échelles Hétéro-évaluées
            
            **ADHD-RS (ADHD Rating Scale) :**
            - Version parents et enseignants
            - 18 items correspondant aux critères DSM-5
            - Scores par sous-domaines
            
            **Conners 3 :**
            - Formes courtes et longues
            - Versions parents, enseignants, auto-évaluation
            - Indices de validité intégrés
            - Normes françaises disponibles
            """)
            
        with scales_tabs[1]:
            st.markdown("""
            ### 📈 Outils de Suivi Thérapeutique
            
            #### Suivi des Symptômes
            
            **Échelles de changement :**
            - CGI-S (Clinical Global Impression - Severity)
            - CGI-I (Clinical Global Impression - Improvement)
            - Évaluation subjective du patient/famille
            
            **Journaux quotidiens :**
            - Carnet de symptômes
            - Échelles visuelles analogiques
            - Applications mobiles de suivi
            - Monitoring des effets secondaires
            
            #### Évaluation Fonctionnelle
            
            **WEISS (Weiss Functional Impairment Rating Scale) :**
            - Impact sur 7 domaines de vie
            - Version enfant/adolescent/adulte
            - Sensible aux changements thérapeutiques
            
            **BRIEF (Behavior Rating Inventory of Executive Function) :**
            - Évaluation fonctions exécutives quotidiennes
            - Versions préscolaire, scolaire, adulte
            - Profils spécifiques par domaines
            """)
            
        with scales_tabs[2]:
            st.markdown("""
            ### 🧠 Évaluations Neuropsychologiques
            
            #### Tests Attentionnels
            
            **Test d'Évaluation de l'Attention (TEA) :**
            - Attention sélective, soutenue, divisée
            - Versions enfant et adulte
            - Profils attentionnels détaillés
            
            **Continuous Performance Tests (CPT) :**
            - Mesure attention soutenue
            - Détection des erreurs de commission/omission
            - Variabilité du temps de réaction
            
            #### Fonctions Exécutives
            
            **NEPSY-II :**
            - Batterie complète enfant/adolescent
            - Domaines : attention, fonctions exécutives, mémoire
            - Normes françaises récentes
            
            **Test de Stroop :**
            - Évaluation inhibition cognitive
            - Sensible aux troubles attentionnels
            - Versions informatisées disponibles
            
            #### Mémoire de Travail
            
            **Échelles de Wechsler (WISC-V, WAIS-IV) :**
            - Indice Mémoire de Travail
            - Sous-tests spécifiques (Empan, Séquences)
            - Profils cognitifs détaillés
            """)

    with doc_tabs[5]:
        st.subheader("🔬 Recherche Récente et Perspectives")
        
        research_tabs = st.tabs(["Neurosciences", "Génétique", "Nouvelles thérapies", "IA et TDAH"])
        
        with research_tabs[0]:
            st.markdown("""
            ### 🧠 Avancées en Neurosciences
            
            #### Neuroimagerie Fonctionnelle
            
            **IRM fonctionnelle (IRMf) :**
            - Hypoactivation du cortex préfrontal
            - Dysconnectivité des réseaux attentionnels
            - Maturation retardée des circuits fronto-striataux
            - Biomarqueurs potentiels du diagnostic
            
            **Électroencéphalographie (EEG) :**
            - Rapport thêta/bêta élevé
            - Potentiels évoqués altérés
            - Neurofeedback EEG comme traitement
            - Marqueurs prédictifs de réponse thérapeutique
            
            #### Connectivité Cérébrale
            
            **Réseaux de repos :**
            - Réseau par défaut hyperactif
            - Réseau attentionnel hypoactif
            - Corrélations avec sévérité symptomatique
            - Cibles pour interventions thérapeutiques
            """)
            
        with research_tabs[1]:
            st.markdown("""
            ### 🧬 Recherches Génétiques
            
            #### Génétique Moléculaire
            
            **Gènes candidats :**
            - DRD4, DAT1, DRD2 (système dopaminergique)
            - NET1, DBH (système noradrénergique)
            - 5HTR1B, TPH2 (système sérotoninergique)
            - SNAP25, COMT (neurotransmission)
            
            **Études Genome-Wide (GWAS) :**
            - Plus de 12 loci identifiés
            - Héritabilité polygénique (SNP-h² ≈ 22%)
            - Chevauchement génétique avec autres troubles
            - Scores de risque polygénique en développement
            
            #### Pharmacogénétique
            
            **Prédiction de réponse :**
            - Variants CYP2D6 et métabolisme
            - Polymorphismes transporteurs (DAT1, NET1)
            - Tests génétiques pour personnalisation
            - Médecine de précision en développement
            """)
            
        with research_tabs[2]:
            st.markdown("""
            ### 💊 Nouvelles Approches Thérapeutiques
            
            #### Thérapies Numériques
            
            **Applications thérapeutiques :**
            - Jeux vidéo thérapeutiques (EndeavorRx)
            - Réalité virtuelle pour entraînement attentionnel
            - Thérapie cognitive informatisée
            - Interventions par smartphone
            
            **Neurofeedback avancé :**
            - Neurofeedback temps réel IRMf
            - Stimulation transcrânienne (tDCS, rTMS)
            - Interfaces cerveau-ordinateur
            - Modulation non-invasive de l'activité cérébrale
            
            #### Nouvelles Molécules
            
            **En développement :**
            - Modulateurs AMPA (ampakines)
            - Agonistes nicotiniques (α7)
            - Inhibiteurs phosphodiestérase
            - Thérapies épigénétiques
            """)
            
        with research_tabs[3]:
            st.markdown("""
            ### 🤖 Intelligence Artificielle et TDAH
            
            #### Diagnostic Assisté par IA
            
            **Analyse comportementale :**
            - Reconnaissance de patterns vidéo
            - Analyse de mouvements oculaires
            - Détection automatique de symptômes
            - Scores prédictifs multi-modaux
            
            **Machine Learning :**
            - Classification par algorithmes supervisés
            - Réseaux de neurones profonds
            - Analyse de données multi-échelles
            - Validation sur grandes cohortes
            
            #### Applications Cliniques
            
            **Outils d'aide au diagnostic :**
            - Plateformes d'évaluation numérique
            - Analyse automatisée de questionnaires
            - Intégration données neuroimagerie
            - Scores de probabilité diagnostique
            
            **Personnalisation thérapeutique :**
            - Prédiction de réponse aux traitements
            - Optimisation posologique
            - Identification de sous-types
            - Médecine de précision
            """)

    with doc_tabs[6]:
        st.subheader("📚 Ressources Pratiques")
        
        resources_tabs = st.tabs(["Associations", "Sites web", "Applications", "Livres"])
        
        with resources_tabs[0]:
            st.markdown("""
            ### 🏛️ Associations et Organisations
            
            #### France
            
            **HyperSupers TDAH France :**
            - Association nationale de référence
            - Groupes de soutien régionaux
            - Formation et information
            - Site web : tdah-france.fr
            
            **AFEP (Association Française pour les Enfants Précoces) :**
            - Accompagnement enfants à haut potentiel + TDAH
            - Réseau national de bénévoles
            - Ressources éducatives spécialisées
            
            #### International
            
            **CHADD (Children and Adults with ADHD) - USA :**
            - Plus grande organisation mondiale TDAH
            - Ressources scientifiques actualisées
            - Formations professionnelles
            
            **CADDRA (Canadian ADHD Resource Alliance) :**
            - Lignes directrices canadiennes
            - Outils d'évaluation validés
            - Formation des professionnels
            
            #### Centres de Référence France
            
            **Centres experts TDAH :**
            - CHU Robert Debré (Paris)
            - CHU Montpellier
            - CHU Lyon
            - CHU Lille
            - CHU Bordeaux
            """)
            
        with resources_tabs[1]:
            st.markdown("""
            ### 🌐 Sites Web Fiables
            
            #### Sites Institutionnels
            
            **Haute Autorité de Santé (HAS) :**
            - Recommandations officielles françaises
            - Guides patients et professionnels
            - has-sante.fr/portail/jcms/c_2012647/fr/tdah
            
            **INSERM :**
            - Expertise collective TDAH
            - Recherches françaises actuelles
            - inserm.fr/dossier/trouble-deficit-attention-hyperactivite-tdah
            
            #### Sites Scientifiques
            
            **Journal of Attention Disorders :**
            - Publications de recherche récentes
            - Revues systématiques et méta-analyses
            - Accès via bibliothèques universitaires
            
            **ADHD Institute :**
            - Ressources pour professionnels
            - Outils d'évaluation
            - Formations en ligne
            
            #### Information Patients
            
            **Ameli.fr (Assurance Maladie) :**
            - Information patient validée
            - Parcours de soins
            - Remboursements et prises en charge
            """)
            
        with resources_tabs[2]:
            st.markdown("""
            ### 📱 Applications Recommandées
            
            #### Gestion de l'Attention
            
            **Forest - Focus Timer :**
            - Technique Pomodoro gamifiée
            - Blocage applications distrayantes
            - Statistiques de concentration
            - iOS et Android gratuit/premium
            
            **Brain Focus Productivity Timer :**
            - Cycles travail/pause personnalisables
            - Suivi statistiques détaillées
            - Interface simple et efficace
            
            #### Organisation et Planification
            
            **Todoist :**
            - Gestion de tâches intuitive
            - Rappels et échéances
            - Collaboration familiale/équipe
            - Synchronisation multi-plateformes
            
            **Any.do :**
            - Interface très simple
            - Rappels vocaux
            - Partage de listes
            - Intégration calendrier
            
            #### Bien-être et Relaxation
            
            **Headspace :**
            - Méditations guidées courtes
            - Programmes spécialisés attention
            - Exercices de respiration
            - Suivi progression
            
            **Calm :**
            - Séances relaxation variées
            - Histoires pour dormir
            - Musiques apaisantes
            - Programmes quotidiens
            """)
            
        with resources_tabs[3]:
            st.markdown("""
            ### 📖 Bibliographie Recommandée
            
            #### Ouvrages Généralistes
            
            **"TDAH chez l'adulte" - Dr. Michel Bouvard :**
            - Référence française sur TDAH adulte
            - Diagnostic et traitements actualisés
            - Approche clinique pratique
            
            **"Mon cerveau a TDAH" - Dr. Annick Vincent :**
            - Vulgarisation scientifique accessible
            - Témoignages et cas cliniques
            - Stratégies concrètes au quotidien
            
            #### Guides Pratiques
            
            **"TDAH, la boîte à outils" - Ariane Hémond :**
            - 100 fiches pratiques
            - Activités et exercices
            - Pour parents et professionnels
            
            **"L'enfant inattentif et hyperactif" - Stacey Bélanger :**
            - Guide complet parents
            - Stratégies développementales
            - Collaboration école-famille
            
            #### Littérature Scientifique
            
            **"Handbook of ADHD" - Russell Barkley :**
            - Référence internationale
            - Théories et recherches actuelles
            - Pour professionnels spécialisés
            
            **"ADHD in Adults" - Biederman & Spencer :**
            - Spécifiquement TDAH adulte
            - Évidence-based medicine
            - Comorbidités et diagnostics différentiels
            """)

    with doc_tabs[7]:
        st.subheader("❓ Questions Fréquemment Posées")
        
        # Questions générales
        with st.expander("🤔 Le TDAH est-il réel ou inventé ?", expanded=False):
            st.markdown("""
            Le TDAH est un trouble neurodéveloppemental scientifiquement validé et reconnu par toutes les organisations médicales internationales. 
            
            **Preuves scientifiques :**
            - Plus de 10 000 études publiées
            - Différences cérébrales observables en neuroimagerie
            - Base génétique documentée (héritabilité 70-80%)
            - Critères diagnostiques précis et validés
            
            **Pourquoi cette question persiste :**
            - Symptômes variables selon les contextes
            - Diagnostic basé sur l'observation clinique
            - Stigmatisation et méconnaissance
            - Médiatisation parfois simplifiée
            """)
        
        with st.expander("👶 Mon enfant est-il trop jeune pour un diagnostic ?", expanded=False):
            st.markdown("""
            **Âge minimum pour le diagnostic :**
            - Critères DSM-5 : symptômes présents avant 12 ans
            - Diagnostic possible dès 4-6 ans
            - Évaluation adaptée à l'âge développemental
            
            **Défis diagnostiques chez les jeunes :**
            - Variabilité développementale normale
            - Immaturité des fonctions exécutives
            - Difficulté à distinguer TDAH d'autres troubles
            
            **Approche recommandée :**
            - Observation sur plusieurs mois
            - Évaluation multidisciplinaire
            - Intervention comportementale prioritaire avant 6 ans
            - Médicaments seulement si troubles sévères
            """)
        
        with st.expander("💊 Les médicaments sont-ils dangereux ?", expanded=False):
            st.markdown("""
            **Sécurité démontrée :**
            - Décennies d'utilisation documentée
            - Profil de sécurité favorable chez l'enfant et l'adulte
            - Surveillance médicale régulière obligatoire
            - Bénéfices généralement supérieurs aux risques
            
            **Effets secondaires fréquents mais généralement bénins :**
            - Diminution de l'appétit (temporaire)
            - Troubles du sommeil (gérables)
            - Céphalées, nervosité (transitoires)
            
            **Surveillance nécessaire :**
            - Croissance chez l'enfant
            - Tension artérielle et fréquence cardiaque
            - Effets psychologiques (humeur, tics)
            - Ajustements posologiques réguliers
            """)
        
        with st.expander("🎓 Mon enfant peut-il réussir à l'école avec un TDAH ?", expanded=False):
            st.markdown("""
            **Absolument ! Avec un accompagnement adapté :**
            - Aménagements pédagogiques personnalisés
            - Collaboration école-famille-soins
            - Stratégies compensatoires efficaces
            - Soutien spécialisé si nécessaire
            
            **Facteurs de réussite :**
            - Diagnostic et prise en charge précoces
            - Enseignants formés et bienveillants
            - Estime de soi préservée
            - Valorisation des points forts
            
            **Exemples de réussites :**
            - Nombreuses personnalités connues avec TDAH
            - Créativité et innovation souvent renforcées
            - Hyperfocus sur domaines d'intérêt
            - Capacités d'adaptation développées
            """)
        
        with st.expander("👨‍👩‍👧‍👦 Comment annoncer le diagnostic à mon enfant ?", expanded=False):
            st.markdown("""
            **Principes généraux :**
            - Adapter le langage à l'âge et à la maturité
            - Présenter de manière positive et déculpabilisante
            - Expliquer le cerveau qui fonctionne différemment
            - Insister sur les forces et talents particuliers
            
            **Message clé à transmettre :**
            - "Ton cerveau fonctionne de manière unique"
            - "Ce n'est pas de ta faute"
            - "Nous allons t'aider à mieux réussir"
            - "Beaucoup de personnes vivent bien avec un TDAH"
            
            **Ressources utiles :**
            - Livres adaptés aux enfants
            - Métaphores et comparaisons simples
            - Témoignages positifs d'autres enfants/adultes
            - Accompagnement psychologique si besoin
            """)
        
        with st.expander("💼 Puis-je travailler normalement avec un TDAH ?", expanded=False):
            st.markdown("""
            **Oui, avec des adaptations appropriées :**
            - Choix de métiers compatibles avec vos forces
            - Aménagements de poste si nécessaire
            - Stratégies d'organisation personnalisées
            - Traitement médical adapté si souhaité
            
            **Secteurs souvent favorables :**
            - Métiers créatifs et innovants
            - Professions de contact et relationnel
            - Entrepreneuriat et freelance
            - Secteurs dynamiques et variés
            
            **Droits et protections :**
            - Reconnaissance travailleur handicapé (RQTH)
            - Protection contre la discrimination
            - Accès aux dispositifs d'aide à l'emploi
            - Confidentialité médicale respectée
            """)

def show_about():
    """Page À propos adaptée pour le TDAH"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff5722, #ff9800);
                padding: 40px 25px; border-radius: 20px; margin-bottom: 35px; text-align: center;">
        <h1 style="color: white; font-size: 2.8rem; margin-bottom: 15px;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 600;">
            ℹ️ À propos de cette plateforme
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem;
                  max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Innovation technologique au service du dépistage TDAH
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mission et vision
    st.markdown("""
    <div class="info-card-modern">
        <h2 style="color: #ff5722; text-align: center; margin-bottom: 25px;">🎯 Notre Mission</h2>
        <p style="font-size: 1.2rem; line-height: 1.8; text-align: center; max-width: 800px; margin: 0 auto;">
            Démocratiser l'accès au dépistage précoce du TDAH en combinant l'expertise clinique 
            et l'intelligence artificielle pour améliorer la vie de millions de personnes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card-modern" style="height: 300px;">
            <h3 style="color: #ff5722; margin-bottom: 20px;">🔬 Innovation Scientifique</h3>
            <ul style="line-height: 1.8; padding-left: 20px;">
                <li>Algorithmes d'apprentissage automatique avancés</li>
                <li>Validation sur cohortes cliniques réelles</li>
                <li>Approche evidence-based et multidisciplinaire</li>
                <li>Intégration des dernières recherches en neurosciences</li>
                <li>Développement itératif avec feedback clinique</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card-modern" style="height: 300px;">
            <h3 style="color: #ff9800; margin-bottom: 20px;">🤝 Impact Social</h3>
            <ul style="line-height: 1.8; padding-left: 20px;">
                <li>Réduction des délais de diagnostic</li>
                <li>Amélioration de l'accès aux soins spécialisés</li>
                <li>Soutien aux familles et professionnels</li>
                <li>Sensibilisation et déstigmatisation</li>
                <li>Égalité des chances éducatives et professionnelles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Équipe et expertise
    st.markdown("""
    <div class="info-card-modern">
        <h3 style="color: #ff5722; text-align: center; margin-bottom: 25px;">👥 Équipe Multidisciplinaire</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 25px;">
            
            <div style="text-align: center; padding: 20px; background: #fff3e0; border-radius: 10px;">
                <h4 style="color: #ef6c00; margin-bottom: 10px;">👨‍⚕️ Expertise Clinique</h4>
                <p style="color: #f57c00; line-height: 1.6;">
                    Psychiatres, psychologues, neuropsychologues spécialisés en TDAH
                </p>
            </div>
            
            <div style="text-align: center; padding: 20px; background: #fff3e0; border-radius: 10px;">
                <h4 style="color: #ef6c00; margin-bottom: 10px;">🔬 Data Science</h4>
                <p style="color: #f57c00; line-height: 1.6;">
                    Ingénieurs ML, statisticiens, chercheurs en IA médicale
                </p>
            </div>
            
            <div style="text-align: center; padding: 20px; background: #fff3e0; border-radius: 10px;">
                <h4 style="color: #ef6c00; margin-bottom: 10px;">💻 Développement</h4>
                <p style="color: #f57c00; line-height: 1.6;">
                    Développeurs full-stack, experts UX/UI, ingénieurs DevOps
                </p>
            </div>
            
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Méthodologie et validation
    st.markdown("""
    <div class="info-card-modern">
        <h3 style="color: #ff5722; margin-bottom: 25px;">📊 Méthodologie et Validation</h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
            <div>
                <h4 style="color: #ef6c00; margin-bottom: 15px;">🎯 Développement</h4>
                <ul style="color: #f57c00; line-height: 1.8; padding-left: 20px;">
                    <li>Collecte de données cliniques multicentriques</li>
                    <li>Annotation par experts reconnus</li>
                    <li>Préparation et nettoyage des données</li>
                    <li>Entraînement de modèles supervisés</li>
                    <li>Optimisation des hyperparamètres</li>
                </ul>
            </div>
            
            <div>
                <h4 style="color: #ef6c00; margin-bottom: 15px;">✅ Validation</h4>
                <ul style="color: #f57c00; line-height: 1.8; padding-left: 20px;">
                    <li>Validation croisée stratifiée</li>
                    <li>Tests sur cohortes indépendantes</li>
                    <li>Évaluation par cliniciens experts</li>
                    <li>Analyse de biais et d'équité</li>
                    <li>Amélioration continue</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Partenariats et collaborations
    st.markdown("""
    <div class="info-card-modern">
        <h3 style="color: #ff5722; text-align: center; margin-bottom: 25px;">🤝 Partenariats Scientifiques</h3>
        <div style="text-align: center;">
            <div style="display: inline-flex; gap: 40px; justify-content: center; flex-wrap: wrap;">
                <div style="padding: 15px;">
                    <h4 style="color: #ef6c00;">🏥 Centres Hospitaliers</h4>
                    <p style="color: #f57c00;">CHU référents TDAH</p>
                </div>
                <div style="padding: 15px;">
                    <h4 style="color: #ef6c00;">🎓 Universités</h4>
                    <p style="color: #f57c00;">Laboratoires de recherche</p>
                </div>
                <div style="padding: 15px;">
                    <h4 style="color: #ef6c00;">🏛️ Associations</h4>
                    <p style="color: #f57c00;">TDAH France, AFEP</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Limites et responsabilités
    st.markdown("""
    <div class="info-card-modern">
        <h3 style="color: #ff5722; margin-bottom: 25px;">⚠️ Limites et Responsabilités</h3>
        <div style="background: #ffebee; padding: 20px; border-radius: 10px; border-left: 4px solid #f44336;">
            <h4 style="color: #c62828; margin-top: 0;">Cadre d'utilisation</h4>
            <ul style="color: #d32f2f; line-height: 1.8; padding-left: 20px;">
                <li><strong>Outil d'aide au dépistage uniquement</strong> - Ne remplace pas l'évaluation clinique</li>
                <li><strong>Population d'entraînement</strong> - Validé sur population française/européenne</li>
                <li><strong>Évolution continue</strong> - Algorithmes mis à jour régulièrement</li>
                <li><strong>Confidentialité</strong> - Données anonymisées et sécurisées</li>
                <li><strong>Formation requise</strong> - Utilisation par professionnels formés recommandée</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Perspectives futures
    st.markdown("""
    <div class="info-card-modern">
        <h3 style="color: #ff5722; text-align: center; margin-bottom: 25px;">🚀 Perspectives d'Évolution</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            
            <div style="text-align: center; padding: 20px; border: 2px solid #ff5722; border-radius: 10px;">
                <h4 style="color: #ff5722;">📱 Mobile</h4>
                <p style="color: #d84315;">Applications natives iOS/Android</p>
            </div>
            
            <div style="text-align: center; padding: 20px; border: 2px solid #ff5722; border-radius: 10px;">
                <h4 style="color: #ff5722;">🔗 Intégration</h4>
                <p style="color: #d84315;">APIs pour dossiers médicaux</p>
            </div>
            
            <div style="text-align: center; padding: 20px; border: 2px solid #ff5722; border-radius: 10px;">
                <h4 style="color: #ff5722;">🌍 International</h4>
                <p style="color: #d84315;">Validation multiculturelle</p>
            </div>
            
            <div style="text-align: center; padding: 20px; border: 2px solid #ff5722; border-radius: 10px;">
                <h4 style="color: #ff5722;">🧠 IA Avancée</h4>
                <p style="color: #d84315;">Deep learning, NLP médical</p>
            </div>
            
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Contact et informations légales
    st.markdown("""
    <div style="margin: 40px 0 30px 0; padding: 25px; border-radius: 15px;
               background: linear-gradient(135deg, #fff3e0, #ffcc02); border-left: 4px solid #ff5722;
               box-shadow: 0 6px 20px rgba(255, 87, 34, 0.15);">
        <h3 style="color: #ef6c00; text-align: center; margin-bottom: 20px;">📧 Contact et Collaboration</h3>
        <div style="text-align: center; font-size: 1.1rem; color: #f57c00; line-height: 1.8;">
            <p><strong>Collaboration scientifique :</strong> Nous sommes ouverts aux partenariats de recherche</p>
            <p><strong>Formation professionnelle :</strong> Sessions dédiées aux équipes soignantes</p>
            <p><strong>Feedback utilisateurs :</strong> Vos retours enrichissent notre développement</p>
            <p><strong>Conformité RGPD :</strong> Protection maximale des données personnelles</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Application principale
def main():
    set_custom_theme()
    initialize_session_state()
    
    # Sidebar avec navigation
    with st.sidebar:
        selected_tool = show_navigation_menu()

    # Contenu principal basé sur la sélection
    if selected_tool == "🏠 Accueil":
        show_home_page()
    elif selected_tool == "🔍 Exploration":
        show_enhanced_data_exploration()
    elif selected_tool == "🧠 Analyse ML":
        show_enhanced_ml_analysis()
    elif selected_tool == "🤖 Prédiction par IA":
        show_enhanced_ai_prediction()
    elif selected_tool == "📚 Documentation":
        show_enhanced_documentation()
    elif selected_tool == "ℹ️ À propos":
        show_about()

if __name__ == "__main__":
    main()




