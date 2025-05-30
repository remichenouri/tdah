# -*- coding: utf-8 -*-
"""
Application Streamlit optimisée pour le dépistage TDAH
Corrigée et optimisée selon les meilleures pratiques
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.impute import SimpleImputer

import joblib
import requests
from io import BytesIO
import warnings
import os
import time
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configuration optimisée de la page
st.set_page_config(
    page_title="Dépistage TDAH - IA Avancée",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': "Application de dépistage TDAH utilisant l'intelligence artificielle"
    }
)

# Initialisation optimisée du session state
def init_session_state():
    """Initialise les variables de session de manière optimisée"""
    default_values = {
        'asrs_responses': {},
        'last_topic': 'X',
        'run': False,
        'model': None,
        'data_loaded': False,
        'models_trained': False
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Style CSS amélioré et optimisé
def load_css():
    """Charge les styles CSS de manière optimisée"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            color: #1a237e;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.8rem;
            color: #3949ab;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e3f2fd;
            padding-bottom: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #1976d2;
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .warning-box {
            background: linear-gradient(145deg, #fff3e0, #ffe0b2);
            border: 2px solid #ff9800;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(255, 152, 0, 0.2);
        }
        .success-box {
            background: linear-gradient(145deg, #e8f5e8, #c8e6c8);
            border: 2px solid #4caf50;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
        }
        .info-box {
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            border: 2px solid #2196f3;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
        }
        .stProgress > div > div > div > div {
            background-color: #1976d2;
        }
        .error-container {
            background: linear-gradient(145deg, #ffebee, #ffcdd2);
            border: 2px solid #f44336;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# =================== FONCTIONS UTILITAIRES OPTIMISÉES ===================

@st.cache_data(ttl=3600, show_spinner="Chargement des données...", persist="disk")
def load_data():
    """Charge les données avec cache optimisé et gestion d'erreurs robuste"""
    try:
        logger.info("Tentative de chargement des données depuis Google Drive")
        file_id = '1FYfOf9VT9lymHxlxjiGvuy-UdoddcV8P'
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Session optimisée avec retry
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = session.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Gestion des avertissements de téléchargement Google Drive
                if 'download_warning' in response.cookies:
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            confirm_token = value
                            response = session.get(f'{url}&confirm={confirm_token}', timeout=30)
                            response.raise_for_status()
                            break
                
                # Lecture avec gestion d'encodage améliorée
                content = BytesIO(response.content)
                
                # Tentative avec différents encodages et séparateurs
                encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
                separators = [',', ';', '\t']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            content.seek(0)
                            df = pd.read_csv(content, encoding=encoding, sep=sep, engine='python')
                            if len(df.columns) > 1 and len(df) > 0:
                                logger.info(f"Données chargées avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
                                st.session_state.data_loaded = True
                                return df
                        except Exception as e:
                            logger.debug(f"Échec avec encoding {encoding}, sep {sep}: {e}")
                            continue
                
                break
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Backoff exponentiel
        
        # Si le chargement échoue, créer des données de démonstration
        logger.warning("Chargement depuis Google Drive échoué, création de données de démonstration")
        return create_demo_dataset()
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        st.error(f"Erreur de chargement : {str(e)}")
        return create_demo_dataset()

@st.cache_data(ttl=3600)
def create_demo_dataset():
    """Crée un jeu de données de démonstration optimisé"""
    try:
        np.random.seed(42)
        n = 1000  # Dataset plus large pour de meilleurs tests
        
        # Génération de données réalistes
        age = np.random.normal(35, 12, n).clip(10, 70).astype(int)
        genre = np.random.choice(['Homme', 'Femme'], n, p=[0.6, 0.4])  # Prévalence réelle TDAH
        
        # Scores corrélés de manière réaliste
        base_inattention = np.random.beta(2, 3, n) * 10
        base_hyperactivite = np.random.beta(2, 4, n) * 10
        base_impulsivite = np.random.beta(2, 4, n) * 10
        
        # Ajout de corrélations réalistes
        inattention_score = base_inattention + np.random.normal(0, 1, n)
        hyperactivite_score = base_hyperactivite + 0.6 * base_inattention + np.random.normal(0, 1, n)
        impulsivite_score = base_impulsivite + 0.4 * base_hyperactivite + np.random.normal(0, 1, n)
        
        # Limitation des scores
        inattention_score = np.clip(inattention_score, 1, 10)
        hyperactivite_score = np.clip(hyperactivite_score, 1, 10)
        impulsivite_score = np.clip(impulsivite_score, 1, 10)
        
        # Génération du diagnostic basé sur les scores (logique réaliste)
        total_score = inattention_score + hyperactivite_score + impulsivite_score
        probability_tdah = 1 / (1 + np.exp(-(total_score - 18) / 3))  # Logistique
        tdah = np.random.binomial(1, probability_tdah, n)
        tdah_labels = ['Oui' if x == 1 else 'Non' for x in tdah]
        
        # Données supplémentaires
        niveau_etudes = np.random.choice(
            ['Primaire', 'Collège', 'Lycée', 'Université', 'Post-universitaire'], 
            n, p=[0.1, 0.15, 0.25, 0.35, 0.15]
        )
        
        data = {
            'Age': age,
            'Genre': genre,
            'Inattention_Score': inattention_score,
            'Hyperactivite_Score': hyperactivite_score,
            'Impulsivite_Score': impulsivite_score,
            'Niveau_Etudes': niveau_etudes,
            'TDAH': tdah_labels
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Dataset de démonstration créé: {len(df)} lignes")
        st.info("ℹ️ Données de démonstration chargées (1000 échantillons)")
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du dataset de démonstration: {e}")
        # Dataset minimal en cas d'erreur
        return pd.DataFrame({
            'Age': [25, 30, 35, 40],
            'Genre': ['Homme', 'Femme', 'Homme', 'Femme'],
            'Inattention_Score': [5.0, 7.0, 3.0, 8.0],
            'Hyperactivite_Score': [4.0, 6.0, 2.0, 7.0],
            'Impulsivite_Score': [3.0, 8.0, 2.0, 6.0],
            'TDAH': ['Non', 'Oui', 'Non', 'Oui']
        })

@st.cache_data(persist="disk")
def advanced_preprocessing(df, target_column='TDAH'):
    """Préprocessing avancé avec gestion d'erreurs optimisée"""
    if df is None or df.empty:
        logger.error("DataFrame vide ou None dans preprocessing")
        return None, None

    try:
        df_processed = df.copy()
        feature_info = {'preprocessing_steps': []}

        # 1. Gestion des valeurs manquantes améliorée
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns

        # Imputation numérique avec différentes stratégies
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].skew() > 1:  # Distribution asymétrique
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                feature_info['preprocessing_steps'].append(f"Imputation {col}")

        # Imputation catégorielle
        for col in categorical_cols:
            if col != target_column and df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode()
                if len(mode_value) > 0:
                    df_processed[col].fillna(mode_value[0], inplace=True)
                else:
                    df_processed[col].fillna('Unknown', inplace=True)
                feature_info['preprocessing_steps'].append(f"Imputation {col}")

        # 2. Feature Engineering avancé
        score_columns = [col for col in df_processed.columns if 'score' in col.lower()]
        if len(score_columns) >= 2:
            df_processed['Score_Total'] = df_processed[score_columns].sum(axis=1)
            df_processed['Score_Moyen'] = df_processed[score_columns].mean(axis=1)
            df_processed['Score_Std'] = df_processed[score_columns].std(axis=1)
            df_processed['Score_Max'] = df_processed[score_columns].max(axis=1)
            df_processed['Score_Min'] = df_processed[score_columns].min(axis=1)
            
            # Ratios significatifs
            if 'Inattention_Score' in df_processed.columns and 'Hyperactivite_Score' in df_processed.columns:
                df_processed['Ratio_Inatt_Hyper'] = (
                    df_processed['Inattention_Score'] / 
                    (df_processed['Hyperactivite_Score'] + 0.1)  # Éviter division par zéro
                )
            
            feature_info['engineered_features'] = [
                'Score_Total', 'Score_Moyen', 'Score_Std', 'Score_Max', 'Score_Min'
            ]

        # Binning de l'âge optimisé
        if 'Age' in df_processed.columns:
            df_processed['Age_Group'] = pd.cut(
                df_processed['Age'],
                bins=[0, 12, 18, 25, 35, 50, 100],
                labels=['Enfant', 'Adolescent', 'Jeune_Adulte', 'Adulte', 'Adulte_Mature', 'Senior']
            )
            feature_info['age_groups'] = True

        # 3. Encodage optimisé des variables catégorielles
        categorical_mappings = {}
        for col in categorical_cols:
            if col != target_column:
                try:
                    le = LabelEncoder()
                    # Gestion des valeurs manquantes avant encodage
                    df_processed[col] = df_processed[col].astype(str)
                    df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
                    categorical_mappings[col] = le
                except Exception as e:
                    logger.warning(f"Erreur encodage {col}: {e}")

        # 4. Détection et gestion des outliers
        for col in numeric_cols:
            if col != target_column:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df_processed[col] < lower_bound) | 
                           (df_processed[col] > upper_bound)).sum()
                
                if outliers > 0:
                    logger.info(f"Outliers détectés dans {col}: {outliers}")
                    # Cap des outliers instead of removal
                    df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)

        feature_info['categorical_mappings'] = categorical_mappings
        feature_info['original_shape'] = df.shape
        feature_info['processed_shape'] = df_processed.shape
        feature_info['numeric_features'] = list(numeric_cols)
        feature_info['categorical_features'] = list(categorical_cols)

        logger.info(f"Preprocessing terminé: {df.shape} -> {df_processed.shape}")
        return df_processed, feature_info

    except Exception as e:
        logger.error(f"Erreur lors du preprocessing: {e}")
        return df, {'error': str(e)}

# =================== FONCTIONS MACHINE LEARNING OPTIMISÉES ===================

@st.cache_resource(show_spinner="Entraînement des modèles ML...")
def train_multiple_models(df, target_column='TDAH'):
    """Entraîne plusieurs modèles ML avec optimisation avancée"""
    try:
        if df is None or target_column not in df.columns:
            logger.error(f"DataFrame invalide ou colonne {target_column} manquante")
            return None, None, None, None

        # Préparation des données
        X = df.drop(columns=[target_column])
        y = df[target_column].map({'Oui': 1, 'Non': 0})

        # Nettoyage des données
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 20:  # Seuil minimum augmenté
            logger.error(f"Pas assez de données pour l'entraînement: {len(X)}")
            return None, None, None, None

        # Sélection automatique des features numériques
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            logger.error("Aucune feature numérique trouvée")
            return None, None, None, None

        X_numeric = X[numeric_features]

        # Vérification de la variabilité des features
        X_numeric = X_numeric.loc[:, X_numeric.var() > 1e-8]  # Supprime les features constantes

        if X_numeric.shape[1] == 0:
            logger.error("Aucune feature variable trouvée")
            return None, None, None, None

        # Division stratifiée optimisée
        test_size = min(0.3, max(0.1, 50 / len(X)))  # Adaptation dynamique de la taille de test
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y
        )

        # Standardisation robuste
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Configuration des modèles optimisée
        models_params = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                },
                'use_scaled': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'use_scaled': True
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'use_scaled': True
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                },
                'use_scaled': False
            }
        }

        # Entraînement avec gestion d'erreurs robuste
        results = {}
        best_models = {}

        for name, config in models_params.items():
            try:
                with st.spinner(f"Optimisation {name}..."):
                    # Cross-validation stratifiée
                    cv = StratifiedKFold(n_splits=min(5, len(y_train) // 10), shuffle=True, random_state=42)
                    
                    grid_search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=cv,
                        scoring='roc_auc',
                        n_jobs=-1,
                        error_score='raise'
                    )

                    # Choix des données d'entraînement
                    X_train_model = X_train_scaled if config['use_scaled'] else X_train
                    X_test_model = X_test_scaled if config['use_scaled'] else X_test

                    grid_search.fit(X_train_model, y_train)
                    y_pred = grid_search.predict(X_test_model)
                    y_pred_proba = grid_search.predict_proba(X_test_model)[:, 1]

                    # Calcul des métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc_score = roc_auc_score(y_test, y_pred_proba)
                    except ValueError:
                        # Cas où une seule classe est présente
                        auc_score = 0.5

                    results[name] = {
                        'accuracy': accuracy,
                        'auc_score': auc_score,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'feature_names': X_numeric.columns.tolist()
                    }

                    best_models[name] = grid_search.best_estimator_
                    logger.info(f"Modèle {name} entraîné: AUC={auc_score:.3f}")

            except Exception as e:
                logger.error(f"Erreur entraînement {name}: {e}")
                continue

        if not results:
            logger.error("Aucun modèle n'a pu être entraîné")
            return None, None, None, None

        st.session_state.models_trained = True
        logger.info(f"Entraînement terminé: {len(results)} modèles")
        return results, best_models, scaler, (X_test, y_test)

    except Exception as e:
        logger.error(f"Erreur générale ML: {e}")
        return None, None, None, None

@st.cache_data
def perform_feature_analysis(df, target_column='TDAH'):
    """Analyse optimisée des features avec sélection automatique"""
    try:
        if df is None or target_column not in df.columns:
            return None

        X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
        y = df[target_column].map({'Oui': 1, 'Non': 0})

        # Nettoyage
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) == 0 or X.shape[1] == 0:
            return None

        # Sélection des meilleures features avec gestion d'erreurs
        k = min(10, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        
        try:
            X_selected = selector.fit_transform(X, y)
        except ValueError as e:
            logger.warning(f"Erreur sélection features: {e}")
            return None

        # Calcul des scores avec gestion des valeurs infinies
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        # Remplacement des valeurs infinies/NaN
        scores = np.nan_to_num(scores, nan=0.0, posinf=1000.0, neginf=0.0)
        pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=0.0)

        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores,
            'P_value': pvalues
        }).sort_values('Score', ascending=False)

        return feature_scores

    except Exception as e:
        logger.error(f"Erreur analyse features: {e}")
        return None
        
def main():
    st.title("Mon Application TDAH")
    # Votre code ici

if __name__ == "__main__":
    main()
