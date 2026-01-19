import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import json
import base64
from io import BytesIO, StringIO
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
import itertools
import re

# --- Configurazione Iniziale di Streamlit ---
st.set_page_config(layout="wide", page_title="Analisi Ranking FT (SHAP)")

# --- Inizializzazione Session State ---
# Variabili di stato per i dati caricati e i risultati del modello
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'shap_df' not in st.session_state:
    st.session_state['shap_df'] = None
if 'perimeter' not in st.session_state:
    st.session_state['perimeter'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'shap_data' not in st.session_state:
    st.session_state['shap_data'] = None
if 'training_time' not in st.session_state:
    st.session_state['training_time'] = None # Salva il tempo di esecuzione

# CSS personalizzato per migliorare la visibilità dei tab
st.markdown("""
            <style>
            /* Stile per i tab */
            .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            }
            .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 20px;
            border: 2px solid #e0e0e0;
            }
            .stTabs [aria-selected="true"] {
            background-color: #41a4ff !important;
            color: white !important;
            border-color: #41a4ff !important;
            }
            /* Riduzione spaziatura sottotitolo */
            .main .block-container {
            padding-top: 2rem;
            }
            </style>
""", unsafe_allow_html=True)

st.title("Financial Times Ranking Analysis")
st.markdown("*App to analyze the factors that influence the Financial Times ranking, using an XGBoost model and SHAP explainability*")


# --- Funzioni Utilità per Caricamento Dati ---

# @st.cache_data è rimosso perché il file cambia
def get_sheet_names(uploaded_file):
    """Restituisce i nomi dei fogli di un file Excel caricato."""
    try:
        # Usa pandas.ExcelFile per leggere solo i nomi dei fogli
        xls = pd.ExcelFile(uploaded_file)
        return xls.sheet_names
    except Exception as e:
        st.error(f"Errore nella lettura dei nomi dei fogli: {e}")
        return []

def load_data_from_upload(uploaded_file, sheet_name, invert_rank: bool = True):
    """Carica e pre-processa i dati dal file Excel caricato."""
    if uploaded_file is None or sheet_name is None:
        return None
        
    try:
        # Carica il DataFrame dallo sheet selezionato
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # --- Pulizia e Pre-Elaborazione (Come nel codice originale) ---
        # Cerca colonne per l'eliminazione e la rinomina (resilienza a nomi non esatti)
        # Assumiamo che il file caricato abbia la stessa struttura del file originale
        
        # Identifica le colonne chiave
        col_map = {
            'School name': 'School name',
            'Location, by primary campus': 'Location',
            'Rank': 'Rank'
        }
        
        # Rinomina le colonne chiave
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

        # Controlla la presenza delle colonne minime richieste
        required_cols = ['School name', 'Rank']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Il file Excel deve contenere le colonne: '{' e '.join(col_map.keys())}' o già rinominate in 'School name' e 'Rank'.")
            return None

        # Inversione dei rank (100 = rank #1)
        if invert_rank:
            n_rows = df.shape[0]
            for c in df.columns:
                # Controllo più robusto: applica l'inversione solo se 'rank' è nel nome e se il tipo è numerico
                if 'rank' in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = n_rows - df[c] + 1
            
        # Resetta l'upload_file (necessario per resettare il puntatore e poter rileggere)
        uploaded_file.seek(0)
        
        return df

    except Exception as e:
        st.error(f"Errore durante la pre-elaborazione dei dati: {e}")
        return None
    

# Normalizza il nome delle colonne (per gestire la colorazione dei grafi delle variabili)
def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Tipo di variabili (per gestire la colorazione dei grafi delle variabili)
def build_var_type_map(df: pd.DataFrame) -> dict:
    """
    Ritorna dict: {colonna_originale -> 'Alumni survey' | 'School survey' | 'Unknown'}
    Usa la tua logica di matching su nome normalizzato.
    """
    var_type = {}
    for col in df.columns:
        c = norm_col(col)

        # --- Alumni survey ---
        if 'salary today' in c:
            var_type[col] = 'Alumni survey'
        elif 'weighted salary' in c:
            var_type[col] = 'Alumni survey'
        elif 'salary' in c and 'increase' in c:
            var_type[col] = 'Alumni survey'
        elif 'value for money' in c:
            var_type[col] = 'Alumni survey'
        elif 'career progress' in c:
            var_type[col] = 'Alumni survey'
        elif 'aims achieved' in c:
            var_type[col] = 'Alumni survey'
        elif 'alumni network' in c:
            var_type[col] = 'Alumni survey'
        elif 'career' in c and 'service' in c:
            var_type[col] = 'Alumni survey'
        elif 'international work mobility' in c:
            var_type[col] = 'Alumni survey'

        # --- School survey ---
        elif 'international faculty' in c:
            var_type[col] = 'School survey'
        elif 'female faculty' in c:
            var_type[col] = 'School survey'
        elif 'faculty with doctorates' in c:
            var_type[col] = 'School survey'
        elif 'women on board' in c:
            var_type[col] = 'School survey'
        elif 'international board' in c:
            var_type[col] = 'School survey'
        elif 'female student' in c:
            var_type[col] = 'School survey'
        elif 'international student' in c:
            var_type[col] = 'School survey'
        elif 'employed' in c and 'months' in c:
            var_type[col] = 'School survey'
        elif 'international course experience' in c:
            var_type[col] = 'School survey'
        elif 'esg' in c:
            var_type[col] = 'School survey'
        elif 'carbon footprint' in c:
            var_type[col] = 'School survey'
        else:
            var_type[col] = 'Unknown'

    return var_type

# Resetta tutto quando viene cambiato il toggle
def reset_app_state_on_toggle():
    # Cancella dati e risultati (lascia eventualmente altre preferenze)
    st.session_state['df'] = None
    st.session_state['shap_df'] = None
    st.session_state['perimeter'] = None
    st.session_state['model_results'] = None
    st.session_state['shap_data'] = None
    st.session_state['training_time'] = None

    # Opzionale: resetta anche selezioni UI che dipendono dai dati
    for k in ['sheet_select', 'exp_highlight_unis', 'exp_variable_select',
              'uni_drilldown_select', 'comp_location_filter', 'comp_multiselect',
              'uni_scenario_select', 'rel_corr_method', 'rel_corr_grafo_slider',
              'rel_mi_grafo_slider', 'mod_slider_features']:
        if k in st.session_state:
            del st.session_state[k]

    # Opzionale: elimina tutti gli slider dello scenario (chiavi dinamiche)
    for k in list(st.session_state.keys()):
        if k.startswith("slider_"):
            del st.session_state[k]

# --- Funzione di Training e SHAP (Memorizzata in session_state) ---

# @st.cache_resource è rimosso per permettere il re-training
def train_and_shap(df):
    """Addestra il modello XGBoost e calcola i valori SHAP. Salva in session_state."""
    start_time = time.time()
    random_seed = 3
    
    # Prepara i dati
    # Assumiamo che tutte le colonne numeriche tranne 'Rank' siano feature
    feature_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'Rank']
    # Rimuovi anche 'Location' che è categorica ma a volte può essere interpretata come numerica se ha valori numerici
    feature_cols = [c for c in feature_cols_all if c not in ['School name', 'Location']] 
    
    # Rimuovi righe con NaN nelle colonne feature o target
    df_clean = df.dropna(subset=feature_cols + ['Rank'])
    
    if len(df_clean) == 0:
        st.error("Nessun dato valido rimasto dopo la rimozione dei NaN nelle colonne feature/target.")
        return None

    X = df_clean[feature_cols].values
    y = df_clean['Rank'].values
    
    # 1. Training XGBoost con Cross-Validation
    xgb_model_cv = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=random_seed, objective='reg:squarederror'
    )
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores_r2 = cross_val_score(xgb_model_cv, X, y, cv=kfold, scoring='r2')
    cv_scores_rmse = -cross_val_score(xgb_model_cv, X, y, cv=kfold, scoring='neg_root_mean_squared_error')

    cv_results = {
        'R2_mean': cv_scores_r2.mean(),
        'R2_std': cv_scores_r2.std(),
        'RMSE_mean': cv_scores_rmse.mean(),
        'RMSE_std': cv_scores_rmse.std(),
    }

    # 2. Train Modello Finale su tutto il dataset
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=random_seed, objective='reg:squarederror'
    )
    xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    
    final_r2 = r2_score(y, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y, y_pred))

    final_results = {
        'R2': final_r2,
        'RMSE': final_rmse
    }

    # 3. SHAP ANALYSIS
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    # Crea un DataFrame per i valori SHAP (usando df_clean per gli indici)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, 'School name', df_clean['School name'].values)
    shap_df.insert(1, 'Location', df_clean['Location'].values)
    shap_df.insert(2, 'Rank', df_clean['Rank'].values)
    shap_df.insert(3, 'predicted_score', y_pred)
    
    # Dati necessari per i plot SHAP
    X_with_ranking = np.column_stack([X, y])
    shap_values_extended = np.column_stack([shap_values, np.zeros(len(y))])

    # Calcolo del tempo di esecuzione
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Aggiorna session state con i risultati
    st.session_state['shap_df'] = shap_df
    st.session_state['df'] = df_clean # Aggiorna il DF in session_state con il DF pulito
    st.session_state['model_results'] = {
        'cv': cv_results,
        'final': final_results,
        'feature_cols': feature_cols
    }
    st.session_state['shap_data'] = {
        'model': xgb_model, 
        'explainer': explainer, 
        'shap_values': shap_values, 
        'X': X, 
        'y': y, 
        'X_with_ranking': X_with_ranking, 
        'shap_values_extended': shap_values_extended
    }
    st.session_state['training_time'] = execution_time
    st.success(f"Modello XGBoost addestrato con successo in **{execution_time:.2f} secondi** ⏱️")


def get_lab_and_mode(perimeter: str, invert_rank: bool):
    """
    perimeter: 'Master in Management' | 'Master in Finance'
    invert_rank: toggle (True => Rank invertito)
    """
    if perimeter == 'Master in Management':
        lab = 'MIM'
    elif perimeter == 'Master in Finance':
        lab = 'MIF'
    else:
        # se carichi un file custom e non hai perimeter, decidi cosa fare:
        # a) errore, b) fallback, c) selettore UI
        raise ValueError("Perimeter non riconosciuto. Usa i dati predefiniti MIM/MIF oppure imposta perimeter.")

    mode = 'inverted' if invert_rank else 'default'
    return lab, mode


def get_model_paths(lab: str, mode: str, base_dir: str = "./data"):
    model_path = f"{base_dir}/{lab}_{mode}_xgb_model_final.pkl"
    r2_path = f"{base_dir}/{lab}_{mode}_cv_scores_r2.npy"
    rmse_path = f"{base_dir}/{lab}_{mode}_cv_scores_rmse.npy"
    return model_path, r2_path, rmse_path



def pretrain_and_shap(df, perimeter):
    """Carica il modello pre-addestrato di XGBoost e calcola i valori SHAP. Salva in session_state."""
    start_time = time.time()
    random_seed = 3
    
    # Prepara i dati
    # Assumiamo che tutte le colonne numeriche tranne 'Rank' siano feature
    feature_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'Rank']
    # Rimuovi anche 'Location' che è categorica ma a volte può essere interpretata come numerica se ha valori numerici
    feature_cols = [c for c in feature_cols_all if c not in ['School name', 'Location']] 
    
    # Rimuovi righe con NaN nelle colonne feature o target
    df_clean = df.dropna(subset=feature_cols + ['Rank'])
    
    X = df_clean[feature_cols].values
    y = df_clean['Rank'].values

    # --- Scegli lab/mode in base a perimeter + toggle invert_rank ---
    invert_rank = st.session_state.get("invert_rank", True)
    try:
        lab, mode = get_lab_and_mode(perimeter, invert_rank)
    except ValueError as e:
        st.error(str(e))
        return None

    model_path, r2_path, rmse_path = get_model_paths(lab, mode)

    # --- Carica CV scores ---
    try:
        cv_scores_r2 = np.load(r2_path)
        cv_scores_rmse = np.load(rmse_path)
    except Exception as e:
        st.error(f"Errore nel caricamento dei file CV ({lab}/{mode}): {e}")
        return None


    cv_results = {
        'R2_mean': cv_scores_r2.mean(),
        'R2_std': cv_scores_r2.std(),
        'RMSE_mean': cv_scores_rmse.mean(),
        'RMSE_std': cv_scores_rmse.std(),
    }

    # 2. Caricamento Modello Finale e inferenza su tutto il dataset
    xgb_model = joblib.load(model_path)
    # xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    
    final_r2 = r2_score(y, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y, y_pred))

    final_results = {'R2': final_r2, 'RMSE': final_rmse}

    # 3. SHAP ANALYSIS
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    # Crea un DataFrame per i valori SHAP (usando df_clean per gli indici)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, 'School name', df_clean['School name'].values)
    shap_df.insert(1, 'Location', df_clean['Location'].values)
    shap_df.insert(2, 'Rank', df_clean['Rank'].values)
    shap_df.insert(3, 'predicted_score', y_pred)
    
    # Dati necessari per i plot SHAP
    X_with_ranking = np.column_stack([X, y])
    shap_values_extended = np.column_stack([shap_values, np.zeros(len(y))])

    end_time = time.time() # Registra il tempo finale
    execution_time = end_time - start_time
    
    # Aggiorna session state con i risultati
    st.session_state['shap_df'] = shap_df
    st.session_state['df'] = df_clean # Aggiorna il DF in session_state con il DF pulito
    st.session_state['model_results'] = {
        'cv': cv_results,
        'final': final_results,
        'feature_cols': feature_cols
    }
    st.session_state['shap_data'] = {
        'model': xgb_model, 
        'explainer': explainer, 
        'shap_values': shap_values, 
        'X': X, 
        'y': y, 
        'X_with_ranking': X_with_ranking, 
        'shap_values_extended': shap_values_extended
    }
    st.session_state['training_time'] = execution_time
    st.success(f"Modello XGBoost caricato: **{lab} / {mode}** in **{execution_time:.2f}s** ⏱️")


# --- Funzioni di Plotting (Rimosse decorazioni e logica di caricamento) ---
# Le funzioni di plotting rimangono invariate, ma devono usare le variabili 
# estratte dal `st.session_state`

# Funzione di Plotting Universale per Classifiche con Evidenziazione (OK)
def plot_ranked_variable(df, selected_variable, highlight_unis, title, color_var=None):
    """
    Genera un Plotly Bar Plot per la variabile selezionata, con opzione di evidenziazione.
    """
    
    # Ordina il DataFrame in base alla variabile selezionata (maggiore è meglio)
    df_plot = df.sort_values(selected_variable, ascending=True).reset_index(drop=True)
    
    # Crea la colonna colore per l'evidenziazione
    df_plot['Color'] = df_plot['School name'].apply(lambda x: 'Highlight' if x in highlight_unis else 'Default')

    # Ordine delle categorie per l'asse y
    category_order = df_plot['School name'].tolist()

    # Nuovi colori
    color_discrete_map = {
        'Default': "#d9e5f2",
        'Highlight': '#41a4ff'
    }

    # Crea il grafico
    fig = px.bar(
        df_plot,
        x=selected_variable,
        y='School name',
        orientation='h',
        title=title,
        labels={selected_variable: selected_variable, 'School name': 'Università'},
        color='Color',
        color_discrete_map=color_discrete_map,
        category_orders={'School name': category_order},
        hover_data={'Rank': ':.0f', selected_variable: ':.2f'} # Aggiornato a .0f per il rank
    )
    
    fig.update_layout(
        height=800, 
        yaxis={'categoryorder':'array', 'categoryarray': category_order, 'title': None},
        showlegend=False, # Nascondi la legenda Highlight/Default
        margin=dict(l=10, r=10, t=50, b=50)
    )
    return fig


# Grafici SHAP Generici (Matplotlib/Plotly) (OK)
def plot_shap_summary_bar(shap_values, feature_cols):
    """Grafico SHAP Bar Plot (Feature Importance) - Plotly"""
    
    # Calcola l'importanza media assoluta
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Crea un DataFrame per l'importanza e ordina
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    # Crea il grafico a barre orizzontali con Plotly
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h', 
        labels={'Importance': 'Importanza SHAP (Media Assoluta)', 'Feature': 'Feature'}
    )
    # Aggiorna il layout con un colore più scuro (coerente con la nuova palette)
    fig.update_traces(marker_color='#41a4ff')
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=50),
        height=800,
        yaxis={'title': None} # Rimuove titolo asse Y
    )
    return fig

def plot_shap_beeswarm(shap_values, X, feature_cols):
    """Grafico SHAP Beeswarm Plot - Matplotlib (poi convertito per Streamlit)"""
    
    # Per una visualizzazione ottimale, si usa il plot Matplotlib di SHAP e lo si mostra in Streamlit.
    fig, ax = plt.subplots(figsize=(8, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False, plot_type="dot") 
    plt.tight_layout()
    return fig

def plot_shap_dependence(feat_idx, shap_values_extended, X_with_ranking, feature_cols_with_ranking):
    """Grafico SHAP Dependence Plot per una feature specifica - Matplotlib"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Utilizza 'Rank' come interaction_index per colorare i punti in base al target
    shap.dependence_plot(
        feat_idx, 
        shap_values_extended,  
        X_with_ranking, 
        feature_names=feature_cols_with_ranking,
        interaction_index='Rank',  
        ax=ax,
        show=False
    )
    
    # Rimuovi la feature del ranking_score dai nomi delle feature per il titolo
    feat_name = feature_cols_with_ranking[feat_idx]
    
    plt.title(f"SHAP Dependence Plot: {feat_name}", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

# Grafici SHAP Individuali (Drill-Down) (OK)
def plot_shap_waterfall(uni_idx, explainer, shap_values, X, df, feature_cols):
    """Grafico SHAP Waterfall Plot per una singola università - Matplotlib"""
    
    uni_name = df.iloc[uni_idx]['School name']
    uni_score = df.iloc[uni_idx]['Rank']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crea l'oggetto Explanation richiesto da waterfall_plot
    explanation = shap.Explanation(
        values=shap_values[uni_idx],
        base_values=explainer.expected_value,
        data=X[uni_idx],
        feature_names=feature_cols
    )
    
    # Genera il plot waterfall
    # Nota: shap.waterfall_plot usa una palette interna
    shap.waterfall_plot(explanation, show=False, max_display=12) 
    
    plt.tight_layout()
    return fig

def plot_shap_force(uni_idx, explainer, shap_values, X, feature_cols):
    """Grafico SHAP Force Plot per una singola università - Matplotlib (Embedding)"""
    
    # Crea il plot
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[uni_idx],
        X[uni_idx],
        feature_names=feature_cols,
        matplotlib=False, # Usa la versione HTML/JS
        show=False
    )
    
    # SHAP ha bisogno di Jupyter per renderizzare l'HTML, quindi usiamo un trucco:
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html

# Funzione per l'esportazione dei dati (OK)
def to_excel_download(df_input, df_importance, df_shap):
    """Crea un file Excel in memoria con i tre sheet e restituisce i dati binari."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_input.to_excel(writer, sheet_name='Input Dati', index=False)
        df_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
        df_shap.to_excel(writer, sheet_name='SHAP per Università', index=False)
    processed_data = output.getvalue()
    return processed_data


# --- Struttura dell'App Streamlit (Tabs) ---
tab_esplorazione, tab_modello, tab_drilldown, tab_relazioni, tab_scenario = st.tabs(["Data Upload & Exploration", "Global Model Analysis (SHAP)", "University Drill-Down", "Variable Relationships", "Scenario Analysis"])


# -----------------------------------------------------
# TAB 1: Caricamento & Esplorazione Dati
# -----------------------------------------------------
with tab_esplorazione:
    st.header("Input Data Upload")

    invert_rank = st.toggle(
        "Invert Rank (higher = better position)",
        value=True, 
        key="invert_rank",
        help="If enabled, reverses the values of the 'rank' columns, so that a higher value corresponds to a better position.",
        on_change=reset_app_state_on_toggle
    )
    
    # File Uploader
    col_uploader, col_empty1, col_system_data, col_empty_uploader = st.columns([2, 1, 3, 3])
    with col_uploader:
        st.subheader("⬆️Upload your file") # Nuovo sottotitolo per chiarezza
        st.markdown("Upload your Excel file **(.xlsx)**")

        uploaded_file = st.file_uploader("Upload Excel file", type="xlsx", label_visibility="collapsed")
        df = st.session_state['df']
        sheet_name = None
        
        if uploaded_file is not None:
            # Se un file è caricato, mostra il selettore di sheet
            sheet_names = get_sheet_names(uploaded_file)
            
            if sheet_names:
                default_index = next((i for i, name in enumerate(sheet_names) if '(R)' in name), 0)
                sheet_name = st.selectbox(
                    "Select the worksheet to load:",
                    sheet_names,
                    index=default_index,
                    key='sheet_select'
                )

                invert_rank = st.session_state["invert_rank"]
                    
                # Bottone di caricamento
                if st.button("Upload your file"):
                    # Carica i dati e salva il DF pulito in session_state['df']
                    with st.spinner(f"Uploading data from sheet '{sheet_name}'..."):
                        df_temp = load_data_from_upload(uploaded_file, sheet_name, invert_rank=st.session_state["invert_rank"])
                        if df_temp is not None:
                            st.session_state['df'] = df_temp
                            st.session_state['perimeter'] = ''
                            # Resetta i risultati del modello precedente
                            st.session_state['shap_df'] = None
                            st.session_state['model_results'] = None
                            st.session_state['shap_data'] = None
                            #st.success("Dati caricati e pronti per l'analisi.")
                        else:
                            st.error("Upload failed. Please check the file format.")
                            st.session_state['df'] = None
                    
                    # Forzare un rerun per aggiornare la visualizzazione
                    st.rerun()

    with col_system_data:
        st.subheader("🔄 Load preset data")
        st.markdown("Select a dataset")
        st.markdown("")
        st.markdown("")
        
        col_mim, col_mif = st.columns(2)
        with col_mim:
            if st.button("Preset dataset: MIM 💼", use_container_width=True):
                example_file_path = "./data/FT Master in Management 2025.xlsx"
                with open(example_file_path, "rb") as f:
                    example_data = f.read()
                st.session_state['df'] = load_data_from_upload(BytesIO(example_data), "MIM 2025 (R)", invert_rank=st.session_state["invert_rank"])
                st.session_state['perimeter'] = 'Master in Management'
                # Resetta i risultati del modello precedente
                st.session_state['shap_df'] = None
                st.session_state['model_results'] = None
                st.session_state['shap_data'] = None
        with col_mif:
            if st.button("Preset dataset: MIF 💰", use_container_width=True):
                example_file_path = "./data/FT Masters in Finance 2025.xlsx"
                with open(example_file_path, "rb") as f:
                    example_data = f.read()
                st.session_state['df'] = load_data_from_upload(BytesIO(example_data), "CF25 (R)", invert_rank=st.session_state["invert_rank"])
                st.session_state['perimeter'] = 'Master in Finance'
                # Resetta i risultati del modello precedente
                st.session_state['shap_df'] = None
                st.session_state['model_results'] = None
                st.session_state['shap_data'] = None

    # Contenuto Esplorazione (Mostrato solo se i dati sono stati caricati)
    if st.session_state['df'] is not None:
        st.success("Upload successful.")
        st.markdown("---")
        df = st.session_state['df']
        st.header(f"Input table {st.session_state['perimeter']}")

        # Visualizzazione Tabella Completa
        invert_rank_state = st.session_state.get('invert_rank', True)
        best_rule = "maximum" if invert_rank_state else "minimum"
        best_value = df['Rank'].max() if invert_rank_state else df['Rank'].min()
        st.markdown(f"Data uploaded and pre-processed (Rank: {best_value:.0f} = Rank #1) - {best_rule} value used for the best position.")
        st.markdown(f"*N. of observations:* `{len(df)}` | *N. of variables:* `{len(df.columns)}`")
        st.dataframe(df, use_container_width=True, hide_index=True)
        popover = st.popover("💡 Data dictionary")
        with popover:
            st.caption("Legend:")
            st.markdown("""
            * **School name**: Name of the university.
            * **Location**: Country.
            * **Rank**: Position in the Financial Times ranking (100 = rank #1).
            * **Weighted salary (US$)**: Average weighted salary in USD (three years after completing the Master's) for graduates. Salaries are converted to US dollars using purchasing power parity (PPP) to account for differences in living costs between countries. In addition, weighting is applied to reduce distortion from extreme salaries (high or low).
            * **Salary percentage increase**: Average percentage increase in salary between the initial salary (immediately after completing the Master's) and the current salary (three years later). Often half the weight is on the absolute increase and half on the relative percentage.
            * **Value for money rank**: Ranking that measures the "value for money" of the program, taking into account expected salary, course duration, fees, and opportunity costs (e.g., income lost during the course).
            * **Career progress rank**: Ranking based on the career progression of alumni: changes in seniority, size of the organization they work for now compared to before the Master's, increase in responsibilities, and mobility in roles.
            * **Aims achieved (%)**: Percentage of alumni who report having achieved the professional/personal goals stated at the beginning of the Master's (i.e., how well the course met expectations).
            * **Careers service rank**: Ranking of the business school's career service (placement office, support, network, opportunities offered) as perceived by alumni/data provided.
            * **Alumni network rank**: Ranking of the strength/effectiveness of the alumni network (networking, connections, support, opportunities provided by alumni).
            * **Employed at three months (%)**: Percentage of alumni employed (in a relevant job) within 3 months of completing the Master's.
            * **Female faculty (%)**: Percentage of female faculty members in the school/program.
            * **Female students (%)**: Percentage of female students in the Master's program.
            * **Women on board (%)**: Percentage of female members on the board of the business school.
            * **International faculty (%)**: Percentage of faculty members with non-local/international citizenship (i.e., from different countries).
            * **International students (%)**: Percentage of students with international citizenship (not from the host country) in the program.
            * **International board (%)**: Percentage of board members of the school with international citizenship (not from the host country).
            * **International work mobility rank**: Ranking of international work mobility: measures how many alumni move to a country different from their home country for work after the Master's (changes in employment country).
            * **International course experience rank**: Ranking of international course experience: how much the international component is present in the content, experiences, exchanges, mobility in the course, international projects.
            * **Faculty with doctorates (%)**: Percentage of faculty members who hold a doctorate (PhD) or equivalent.
            * **ESG and net zero teaching rank**: Ranking related to sustainability and ESG (Environmental, Social, Governance), especially for the school's commitment to "net zero" or carbon neutrality goals/green integrated practices.
            * **Carbon footprint rank**: Ranking based on the school's carbon footprint (direct and indirect emissions). Measures how "virtuous" the school is from an environmental perspective.
            """)
        st.markdown("---")
        
        # Controlli per l'evidenziazione
        col_sel, col_empty = st.columns([1, 3])
        with col_sel:
            # Trova l'indice della LUISS (per preselezionarla)
            luiss_name = "Luiss University/Luiss Business School"
            default_unis = [luiss_name] if luiss_name in df['School name'].unique() else df['School name'].unique()[:1].tolist()

            selected_unis = st.multiselect(
                "Select Universities to be highlighted in the charts:",
                sorted(df['School name'].unique()),
                default=default_unis,
                key='exp_highlight_unis'
            )

        # Suddivisione dei grafici a barre in due colonne
        col_rank_chart, col_var_chart = st.columns(2)

        with col_rank_chart:
            # Grafico di Classifica Generale (Rank)
            st.subheader("General Ranking")
            st.markdown("")
            
            fig_rank = plot_ranked_variable(
                df,
                'Rank',
                selected_unis,
                "Ranking by Rank"
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        
        with col_var_chart:
            # Grafico di Classifica per Variabile Selezionata
            st.subheader("Ranking by Selected Variable")

            # Filtra le colonne numeriche disponibili per i plot
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'Rank' and c != 'School name']
            
            selected_variable = st.selectbox(
                "Select the variable to define the ranking:",
                numeric_cols,
                index=numeric_cols.index('Weighted salary (US$)') if 'Weighted salary (US$)' in numeric_cols else 0,
                key='exp_variable_select'
            )
            
            fig_var = plot_ranked_variable(
                df,
                selected_variable,
                selected_unis,
                f"Ranking by: '{selected_variable}'"
            )
            st.plotly_chart(fig_var, use_container_width=True)


# -----------------------------------------------------
# TAB 2: Analisi Globale Modello (SHAP)
# -----------------------------------------------------
with tab_modello:

    st.header("Machine Learning Model and SHAP Calculation")

    if st.session_state['df'] is None:
        st.warning("To train the model, please upload the data in the 'Data Upload & Exploration' tab first.")
    else:
        # Bottone per addestrare il modello
        col_bottone1, col_empty, col_download = st.columns([1, 2, 1])
        with col_bottone1:
            if uploaded_file is not None:
                if st.button("🚀 Start Model Training"):
                    with st.spinner("Training in progress... (this may take a few minutes)"):
                        train_and_shap(st.session_state['df'].copy()) # Pass a copy to avoid side effects
                    # The app will refresh and display the results thanks to the next if
            # If the file has not been uploaded manually, show the training button
            elif uploaded_file is None:
                if st.button("♻️ Load Pre-trained Model"):
                    with st.spinner("Loading model..."):
                        pretrain_and_shap(st.session_state['df'].copy(), st.session_state['perimeter']) # Pass a copy to avoid side effects
                    # The app will refresh and display the results thanks to the next if

        with col_download:
            if st.session_state['shap_df'] is not None:
                shap_df = st.session_state['shap_df']
                df_input = st.session_state['df']
                df_importance = pd.DataFrame({
                    'Feature': st.session_state['model_results']['feature_cols'],
                    'Importance': np.abs(st.session_state['shap_data']['shap_values']).mean(axis=0)
                }).sort_values('Importance', ascending=False)
                excel_data = to_excel_download(df_input, df_importance, shap_df)


                st.info("📊 Download the Excel file with the results 👇")
                st.download_button(
                    label="⬇️ Download file (.xlsx)",
                    data=excel_data,
                    file_name="analisi_ranking_ft.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # Contenuto del Tab 2 (Mostrato solo se il modello è stato addestrato)
        if st.session_state['shap_df'] is not None:
            
            # Estrai dati da session state
            df = st.session_state['df']
            shap_df = st.session_state['shap_df']
            cv_results = st.session_state['model_results']['cv']
            final_results = st.session_state['model_results']['final']
            feature_cols = st.session_state['model_results']['feature_cols']
            shap_data = st.session_state['shap_data']
            
            shap_values = shap_data['shap_values']
            X = shap_data['X']
            X_with_ranking = shap_data['X_with_ranking']
            shap_values_extended = shap_data['shap_values_extended']
            
            feature_cols_with_ranking = feature_cols + ['Rank']
            
            st.markdown("---")
            st.header("Model Estimation")

            # --- Contenitore 1: Riepilogo Dati e Performance ---
            col_data, col_model = st.columns(2)
            
            with col_data:
                st.subheader("Input Data Summary")
                st.markdown(f"* **Number of Universities:** `{len(df)}`")
                st.markdown(f"* **Number of Features:** `{len(feature_cols)}`")
                st.dataframe(df[feature_cols].describe().T.style.format('{:.2f}'), use_container_width=True)
            
            with col_model:
                st.subheader("XGBoost Model Performance")
                st.markdown(f"""
                The model was trained to predict the **Rank**.
                * **R²:** `{final_results['R2']:.4f}`
                * **RMSE:** `{final_results['RMSE']:.4f}`
                * **Mean R² (Cross-Validation):** `{cv_results['R2_mean']:.4f} (+/- {cv_results['R2_std']:.4f})`
                * **Mean RMSE (Cross-Validation):** `{cv_results['RMSE_mean']:.4f} (+/- {cv_results['RMSE_std']:.4f})`
                """)
                st.info("Note: The high R² suggests a good fit for the SHAP interpretability analysis.")

            # --- Contenitore 2: Grafici di Importanza e Distribuzione SHAP ---
            st.markdown("---")
            st.header("SHAP Analysis - Model Interpretability")
            col_bar, col_beeswarm = st.columns(2)

            with col_bar:
                st.subheader("Feature Importance")
                st.caption("Shows the features ranked in descending order of their mean absolute importance in determining the Rank.")
                # Grafico a barre con Plotly
                st.plotly_chart(plot_shap_summary_bar(shap_values, feature_cols), use_container_width=True)

            with col_beeswarm:
                st.subheader("SHAP Beeswarm Plot - Distribution of Effects")
                st.caption("Each point represents a university. The color indicates the feature value, and the horizontal position indicates the impact on the Rank (SHAP value).")
                # Beeswarm plot (Matplotlib, uses cache)
                st.pyplot(plot_shap_beeswarm(shap_values, X, feature_cols), use_container_width=True)

            # --- Contenitore 3: Dependence Plots (Focus sulle interazioni) ---
            st.markdown("---")
            
            # 1. Slider per la selezione del numero di feature
            col_slider, col_empty = st.columns([3, 7]) # Divisione in colonne per limitare la larghezza dello slider
            
            with col_slider:
                max_features = len(shap_values[0])
                top_n_var = st.slider("Number of features to display:", 1, max_features, min(9, max_features), step=1, key='mod_slider_features')

            st.subheader(f"SHAP Dependence Plots (Interactions with Rank) - Top {top_n_var} Features")
            st.caption("These plots show the effect of a single feature on the Rank, coloring the points based on the actual Rank of that university.")

            # Find the indices of the top_n_var most important features
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-top_n_var:][::-1]

            # Determine the layout of the subplots based on the number of top_n_var
            n_cols = 3

            # Create columns for the dependence plots
            all_cols = st.columns(n_cols)

            # Iterate over the top N indices and generate the plots
            for i, feat_idx in enumerate(top_features_idx):
                with all_cols[i % n_cols]:
                    fig = plot_shap_dependence(feat_idx, shap_values_extended, X_with_ranking, feature_cols_with_ranking)
                    st.pyplot(fig, use_container_width=True)
        else:
            if uploaded_file is None:
                st.info("Click the button to upload the pre-trained model and compute the SHAP values.")
            else:
                st.info("Click the button to train the model and compute the SHAP values.")


# -----------------------------------------------------
# TAB 3: Drill-Down per Università
# -----------------------------------------------------
with tab_drilldown:
    
    if st.session_state['shap_df'] is None:
        st.warning("For the Drill-Down analysis, you must first train the model in the 'Global Model Analysis (SHAP)' Tab.")
    else:
        # Extract data from session state
        df = st.session_state['df']
        # Use a version with reset index to align indices with X and shap_values
        df_reset = df.reset_index(drop=True) 
        
        shap_df = st.session_state['shap_df']
        feature_cols = st.session_state['model_results']['feature_cols']
        shap_data = st.session_state['shap_data']
        explainer = shap_data['explainer']
        shap_values = shap_data['shap_values']
        X = shap_data['X']
        
        col_select_uni, col_select_empty = st.columns([2, 5])

        with col_select_uni:
            # Selezione dell'università tramite menu a tendina
            uni_names = sorted(df['School name'].unique())
            
            luiss_name = "Luiss University/Luiss Business School"
            default_index = list(uni_names).index(luiss_name) if luiss_name in uni_names else 0

            selected_uni_name = st.selectbox(
                "Select University:",
                uni_names,
                index=default_index,
                key='uni_drilldown_select' # Added key for Streamlit
            )

        # Extract data for the selected university (for the initial Drill-Down)
        # Note: we use the index in df_reset to access X and shap_values
        uni_idx_in_X_shap = df_reset.query(f"`School name` == '{selected_uni_name}'").index[0]
        uni_score = df_reset.iloc[uni_idx_in_X_shap]['Rank']
        ft_rank = int(df_reset['Rank'].max() - uni_score + 1)

        st.header(f"**{selected_uni_name}** - Score: {uni_score:.0f} (Rank #{ft_rank})")

        col_waterfall, col_force = st.columns([2, 4])
        with col_waterfall:
            st.subheader(f"Waterfall Plot")
            st.caption("Shows how the values of each feature push the model's prediction from the average (`Base Value`) to the predicted score (`f(x)`).")
            # Waterfall Plot
            fig_waterfall = plot_shap_waterfall(uni_idx_in_X_shap, explainer, shap_values, X, df_reset, feature_cols)
            st.pyplot(fig_waterfall, use_container_width=True)

        with col_force:
            # --- BLOCCO 1: Force Plot (Grafico) ---
            st.subheader("Force Plot")
            st.caption("The features in **red** increase the predicted score (`f(x)`), while those in **blue** decrease it.")

            # Force Plot (HTML for interactivity)
            shap_html_content = plot_shap_force(uni_idx_in_X_shap, explainer, shap_values, X, feature_cols)
            st.components.v1.html(shap_html_content, height=350, scrolling=True)

            # ... (Block 2: Positive/Negative Features) ...
            st.markdown("---")

            # Calculate positive and negative features for the list
            pos_features = [(feature_cols[i], shap_values[uni_idx_in_X_shap][i]) for i in range(len(feature_cols)) if shap_values[uni_idx_in_X_shap][i] > 0]
            neg_features = [(feature_cols[i], shap_values[uni_idx_in_X_shap][i]) for i in range(len(feature_cols)) if shap_values[uni_idx_in_X_shap][i] < 0]
            
            pos_features_sorted = sorted(pos_features, key=lambda x: x[1], reverse=True)
            neg_features_sorted = sorted(neg_features, key=lambda x: x[1])

            col_positivi, col_negativi = st.columns(2)
            
            with col_positivi:
                st.markdown("**📈 Top 5 features that increase the predicted score:**")

                for feat, val in pos_features_sorted[:5]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#ffe6e6;
                            padding:8px;
                            border-radius:5px;
                            margin-bottom:5px;
                            border:1px solid #ffcccc;
                        ">
                            <b>{feat}</b>: +{val:.2f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_negativi:
                st.markdown("**📉 Top 5 features that decrease the predicted score:**")

                for feat, val in neg_features_sorted[:5]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#e6f0ff;
                            padding:8px;
                            border-radius:5px;
                            margin-bottom:5px;
                            border:1px solid #cce0ff;
                        ">
                            <b>{feat}</b>: {val:.2f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # --- Sezione Confronto Competitor ---
        st.markdown("---")
        st.header("Competitor Analysis")
        st.caption("Select competitors to view their *Waterfall Plot* and compare their feature values with the reference University.")

        # --- Controls for Competitors ---
        col_filter_loc, col_select_uni_comp, col_empty_sel = st.columns([1, 2, 3])

        # Option 1: Filter by Geographic Area
        locations = ['All geographic areas'] + list(sorted(df['Location'].unique()))
        with col_filter_loc:
            selected_location = st.selectbox("Filter by Geographic Area:", locations, key='comp_location_filter')

        # Apply geographic filter before multi-selection
        df_filtered = df[df['Location'] == selected_location] if selected_location != 'All geographic areas' else df

        # Remove the reference university from the list of competitors
        competitor_uni_names = sorted(df_filtered.query(f"`School name` != '{selected_uni_name}'")['School name'].unique())
        
        with col_select_uni_comp:
            # Option 2: Multi-selection of competitors
            selected_competitors = st.multiselect(
                "Select one or more competitors for comparison:",
                competitor_uni_names,
                key='comp_multiselect'
            )

        # --- Competitor Analysis Report and Plot ---
        if selected_competitors:

            # --- 1. Comparison Table (Forced Order) ---
            st.subheader("Comparison Table (Reference University vs Selected Competitors)")

            # List of universities to compare
            uni_names_to_compare = [selected_uni_name] + selected_competitors

            # Filter data for selected universities
            df_comparison_temp = df[df['School name'].isin(uni_names_to_compare)].drop_duplicates(subset=['School name'])

            # Set index to 'School name'
            df_comparison = df_comparison_temp.set_index('School name', verify_integrity=True)

            # Reorder columns based on the sorted list uni_names_to_compare
            df_comparison = df_comparison.reindex(uni_names_to_compare) # <--- CORREZIONE ORDINE TABELLA

            # Fixed initial columns and features
            initial_cols = ['Rank', 'Location']
            all_model_features = feature_cols
            unique_model_features = [col for col in all_model_features if col not in initial_cols]
            display_cols = initial_cols + unique_model_features

            df_comparison_fixed = df_comparison[display_cols].copy()

            # Apply formatting (ensure columns exist before formatting)
            if 'Rank' in df_comparison_fixed.columns:
                 df_comparison_fixed['Rank'] = df_comparison_fixed['Rank'].apply(lambda x: f"{x:.0f}")
            if 'Weighted salary (US$)' in df_comparison_fixed.columns:
                 df_comparison_fixed['Weighted salary (US$)'] = df_comparison_fixed['Weighted salary (US$)'].apply(lambda x: f"${x:,.0f}")
            
            numeric_feature_cols = [c for c in unique_model_features if pd.api.types.is_numeric_dtype(df_comparison_fixed[c])]
            for col in numeric_feature_cols:
                 df_comparison_fixed[col] = df_comparison_fixed[col].apply(lambda x: f"{x:.2f}")

            # Creazione della tabella trasposta
            df_display = df_comparison_fixed.T
            df_display.columns.name = 'Feature'

            styled_table = df_display.style \
                .set_caption(f"Comparison of Key Data ({selected_uni_name} vs. Competitors)") \
                .set_properties(**{'border-color': 'lightgray'})
                
            st.dataframe(styled_table, use_container_width=True)
            st.markdown("---")

            # --- 2. Waterfall Plot for Each Competitor (Correct Order and Indexing) ---
            st.subheader("Waterfall Plot SHAP for Competitors")
            st.caption("Each plot shows the decomposition of the predicted Rank. The order of the plots reflects the order of the columns in the table.")

            n_cols_waterfall = 4
            # Here we use a fixed number of 4 columns, Streamlit will handle wrapping if there are fewer than 4 plots.
            waterfall_cols = st.columns(n_cols_waterfall)

            # Iterate over the sorted list (uni_names_to_compare)
            for i, competitor_name in enumerate(uni_names_to_compare):

                # Find the correct index in X/SHAP_VALUES using df_reset
                try:
                    comp_idx_in_X_shap = df_reset.query(f"`School name` == '{competitor_name}'").index[0]
                except IndexError:
                    # Handle the case (unlikely if data is consistent) where the name is not found in df_reset
                    st.error(f"Error: University {competitor_name} not found in the cleaned training dataset.")
                    continue

                # Use only available columns (max 4)
                with waterfall_cols[i % n_cols_waterfall]:
                    comp_score = df_reset.iloc[comp_idx_in_X_shap]['Rank']
                    comp_rank = int(df_reset['Rank'].max() - comp_score + 1)
                    st.markdown(f"**{competitor_name}** (Score: {comp_score:.0f}, Rank #{comp_rank})")
                    
                    # Usa comp_idx_in_X_shap per accedere a X e shap_values
                    # e df_reset per accedere alle info generali
                    fig_comp_waterfall = plot_shap_waterfall(comp_idx_in_X_shap, explainer, shap_values, X, df_reset, feature_cols)
                    st.pyplot(fig_comp_waterfall, use_container_width=True)
                    
        else:
            st.info("Select at least one competitor university to start the comparison analysis.")

with tab_relazioni:
    st.markdown("---")
    st.markdown("## 📊 Correlation Analysis")

    if st.session_state['shap_df'] is None:
        st.warning("To perform Drill-Down analysis, you must first train the model in the 'Global Model Analysis (SHAP)' tab.")
    else:
        # Extract data from session state
        df = st.session_state['df']
        # Use a version with reset index to align indices with X and shap_values
        df_reset = df.reset_index(drop=True)

        shap_df = st.session_state['shap_df']
        feature_cols = st.session_state['model_results']['feature_cols']
        shap_data = st.session_state['shap_data']
        explainer = shap_data['explainer']
        shap_values = shap_data['shap_values']
        X = shap_data['X']
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['Rank', 'School name']]

        var_type = build_var_type_map(df)

        col_scelta_corr, col_popover_corr, col_empty = st.columns([2, 2, 6])
        with col_scelta_corr:
            radio_option = st.radio(
                "Correlation Method:",
                ('Pearson', 'Spearman'),
                index=0,
                horizontal=True,
                key='rel_corr_method'
            )
        with col_popover_corr:
            popover_corr = st.popover("ℹ️ Correlation Methods Info")
            with popover_corr:
                st.caption("📘 Correlation Types: Pearson and Spearman")
                st.markdown("""
                - **Pearson Correlation**: Measures the linear relationship between two continuous variables.
                  It assumes the data are normally distributed and is sensitive to outliers. 
                  It’s appropriate when the relationship between variables is linear.

                - **Spearman Correlation**: Measures the monotonic relationship between two variables,
                  based on ranks rather than actual values.
                  It does not require the assumption of normality and is less sensitive to outliers.
                  It is ideal for non-linear but monotonic relationships.
                """)

        col_corr_matrice, col_empty1, col_corr_grafo = st.columns([4, 1, 4])            

        with col_corr_matrice:
            st.markdown("### 🔗 Correlation Matrix")
            method = 'spearman' if radio_option == 'Spearman' else 'pearson'
            corr_matrix = df[numeric_cols].corr(method=method)

            n = len(numeric_cols)
            figsize = max(6, min(0.6 * n, 20))  # scala dinamica tra 6 e 20 pollici
            fig_corr, ax_corr = plt.subplots(figsize=(figsize, figsize))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr, use_container_width=True)

            with st.expander("🔝 Top 10 correlations", expanded=False):
                corr_pairs = (
                    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    .stack()
                    .reset_index()
                    .rename(columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'Correlation'})
                )
                corr_pairs['AbsCorr'] = corr_pairs['Correlation'].abs()
                corr_top = corr_pairs.sort_values('AbsCorr', ascending=False).head(10)

                for _, row in corr_top.iterrows():
                    # Colori dinamici in base al segno della correlazione
                    if row['Correlation'] > 0:
                        color_bg = "#d4f8d4"   # verde chiaro
                        border_color = "#8cd98c"  # verde medio
                        value_bg = "#2E7D32"   # verde scuro per contrasto
                    else:
                        color_bg = "#ffe6e6"   # rosso chiaro
                        border_color = "#ff9999"  # rosso medio
                        value_bg = "#B71C1C"   # rosso scuro per contrasto

                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; justify-content:space-between; 
                                    margin-bottom:5px;">
                            <div style="flex:1; background-color:{color_bg}; padding:6px 2px; border-radius:6px;
                                        border:1px solid {border_color}; color:#1a1a1a;">
                                {row['Feature 1']} ↔ {row['Feature 2']}
                            </div>
                            <div style="width:2px;"></div>
                            <div style="width:70px; text-align:center; 
                                        background-color:{value_bg}; color:white;
                                        padding:6px 0; border-radius:6px;
                                        font-family:monospace;">
                                {row['Correlation']:+.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with col_corr_grafo:
            st.markdown("### 🕸️ Correlation Graph")

            col_corr_slider, col_empty, col_corr_stat = st.columns([2, 1, 1])
            with col_corr_slider:
                soglia_corr = st.slider("Threshold min. Correlation (|r|):", 0.0, 1.0, 0.5, 0.05, key='rel_corr_grafo_slider')

                # Crea grafo
                G_corr = nx.Graph()

                # Aggiungi nodi (una volta sola)
                for feature in numeric_cols:
                    G_corr.add_node(feature)

                # Aggiungi archi basati sulla soglia di correlazione
                corr_edges = 0
                for col1, col2 in itertools.combinations(feature_cols, 2):  # tutte le coppie uniche
                    if abs(corr_matrix.loc[col1, col2]) >= soglia_corr:
                        G_corr.add_edge(col1, col2, weight=corr_matrix.loc[col1, col2], tipo='corr')
                        corr_edges += 1

                # --- Crea rete PyVis ---
                net_corr = Network(
                    height="85vh",
                    width="100%",
                    bgcolor="#ffffff",
                    font_color="black",
                    directed=False
                )

                # Aggiungi i nodi dal grafo NetworkX
                NODE_COLORS = {
                    "Alumni survey": "#87D8F7",  # celeste
                    "School survey": "#0b3d91",  # blu scuro
                    "Unknown": "#81848b",        # grigio
                }

                # for node in G_corr.nodes:
                #     net_corr.add_node(node, label=node, title=f"{node}", color="#41a4ff")
                for node in G_corr.nodes:
                    group = var_type.get(node, "Unknown")
                    net_corr.add_node(
                        node,
                        label=node,  # mostra nome colonna originale
                        title=f"{node}<br>Gruppo: {group}",
                        color=NODE_COLORS.get(group, "#bfc7d5")
    )

                # Aggiungi archi con tooltip che mostra il peso
                for u, v, data in G_corr.edges(data=True):
                    weight = data.get("weight", 0)
                    abs_weight = abs(weight)
                    min_width = 1
                    
                    # Calcola lo spessore basato sulla soglia e sull'intensità
                    if abs_weight >= soglia_corr:
                        # Se soglia_corr è 1.0, questo calcolo fallisce (divisione per zero).
                        if 1.0 - soglia_corr > 0.01: 
                            # Normalizzazione e scalatura da min_width a max_width (es. 10)
                            scaled_width = (abs_weight - soglia_corr) / (1.0 - soglia_corr)
                            edge_width = (scaled_width * 9) + min_width
                        else:
                            # Caso in cui soglia è vicina a 1.0, usiamo uno spessore massimo fisso
                            edge_width = 10 
                    else:
                        # Questo caso è teoricamente impossibile se la logica di NetworkX è corretta
                        edge_width = min_width

                    color = "#F1B17C" if weight >= 0 else "#87D8F7"

                    net_corr.add_edge(u, v, value=abs_weight, title=f"Correlazione: {weight:.3f}", color=color, width=edge_width)

            with col_corr_stat:
                st.caption(f"N. edges: {corr_edges}")


            net_corr.force_atlas_2based()
            html_content = net_corr.generate_html()
            # (inietta JavaScript per centrare e zoomare automaticamente)
            html_content = html_content.replace(
                "</body>",
                """
                <script type="text/javascript">
                    // Attende che la rete sia pronta, poi centra la vista
                    window.addEventListener('load', () => {
                        if (typeof network !== 'undefined') {
                            network.fit();
                        }
                    });
                </script>
                </body>
                """
            )
            st.markdown(
                """
                <br>
                <span style="color:#8fd3ff">●</span> Alumni survey &nbsp; 
                <span style="color:#0b3d91">●</span> School survey &nbsp; 
                <span style="display:inline-block; width:34px; border-top:6px solid #F1B17C; vertical-align:middle;"></span>
                Positive Correlation &nbsp;
                <span style="display:inline-block; width:34px; border-top:6px solid #87D8F7; vertical-align:middle;"></span>
                Negative Correlation
                """,
                unsafe_allow_html=True
            )
            components.html(html_content, height=750, scrolling=True)
            


        st.markdown("---")
        st.markdown("## 💡 Analysis of Relationships through Mutual Information")

        popover_mi = st.popover("ℹ️ Mutual Information Info")
        with popover_mi:
            st.caption("📘 What is Mutual Information (MI)?")
            st.markdown("""
            **Mutual Information (MI)** measures the **statistical dependence** between two variables, 
            assessing how much **information they share**.  
            Unlike **correlation**, MI also captures **non-linear relationships** 
            and is not limited to monotonic relationships.

            **Main Advantages:**
            - Detects non-linear dependencies 📈  
            - Is symmetric: MI(A, B) = MI(B, A) 🔁  
            - Does not require distribution assumptions ⚙️

            The **graph** shows the average amount of information shared between the features of the dataset.
            """)

        col_mi_matrice, col_empty1, col_mi_grafo = st.columns([4, 1, 4]) 

        with col_mi_matrice:
            st.markdown("### 🔗 Mutual Information Across Input Features")
            
            X_cols = pd.DataFrame(X, columns=feature_cols)

            # Inizializza la matrice quadrata dei risultati MI
            mi_matrix = pd.DataFrame(index=X_cols.columns, columns=X_cols.columns, dtype=float)

            # Calcolo della MI per ogni coppia di feature
            for i in range(len(X_cols.columns)):
                for j in range(i, len(X_cols.columns)):
                    feature_A = X_cols.columns[i]
                    feature_B = X_cols.columns[j]

                    if i == j:
                        mi_matrix.loc[feature_A, feature_B] = 0.0
                        mi_matrix.loc[feature_B, feature_A] = 0.0
                    else:
                        mi_score = mutual_info_regression(
                            X_cols[[feature_A]],  # Nx1
                            X_cols[feature_B],    # N
                            random_state=42
                        )[0]
                        # Simmetria della matrice
                        mi_matrix.loc[feature_A, feature_B] = mi_score
                        mi_matrix.loc[feature_B, feature_A] = mi_score

            # --- Visualizza la heatmap ---
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                mi_matrix,
                annot=True,
                cmap="YlGnBu",
                fmt=".2f",
                linewidths=.5,
                annot_kws={"size": 8},
                cbar_kws={'label': 'Mutual Information Score'}
            )
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.clf()


            with st.expander("🔝 Top 10 relationships", expanded=False):
                mi_pairs = (
                    mi_matrix.where(np.triu(np.ones(mi_matrix.shape), k=1).astype(bool))
                    .stack()
                    .reset_index()
                    .rename(columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'MI Score'})
                )

                mi_top = mi_pairs.sort_values('MI Score', ascending=False).head(10)

                for _, row in mi_top.iterrows():
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; justify-content:space-between; 
                                    margin-bottom:5px;">
                            <div style="flex:1; background-color:#edf3f9; padding:6px 2px; border-radius:6px;
                                        border:1px solid ##41a4ff; color:#1a1a1a;">
                                {row['Feature 1']} ↔ {row['Feature 2']}
                            </div>
                            <div style="width:2px;"></div>
                            <div style="width:70px; text-align:center; 
                                        background-color:#41a4ff; color:white;
                                        padding:6px 0; border-radius:6px;
                                        font-family:monospace;">
                                {row['MI Score']:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with col_mi_grafo:
            st.markdown("### 🕸️ Graph of Relationships")

            col_mi_slider, col_empty, col_mi_stat = st.columns([2, 1, 1])
            with col_mi_slider:
                soglia_mi = st.slider("Threshold min. Mutual Information:", 0.0, 1.0, 0.3, 0.05, key='rel_mi_grafo_slider')

            # Crea grafo
            G_mi = nx.Graph()

            # Aggiungi nodi (una volta sola)
            for feature in numeric_cols:
                G_mi.add_node(feature)

            # Aggiungi archi basati sulla soglia
            mi_edges = 0
            for col1, col2 in itertools.combinations(feature_cols, 2):  # tutte le coppie uniche
                if abs(mi_matrix.loc[col1, col2]) >= soglia_mi:
                    G_mi.add_edge(col1, col2, weight=abs(mi_matrix.loc[col1, col2]), tipo='corr')
                    mi_edges += 1

            # --- Crea rete PyVis ---
            net_mi = Network(
                height="85vh",
                width="100%",
                bgcolor="#ffffff",
                font_color="black",
                directed=False
            )

            # Aggiungi i nodi dal grafo NetworkX
            # for node in G_mi.nodes:
            #     net_mi.add_node(node, label=node, title=f"{node}", color="#41a4ff")
            NODE_COLORS = {
                "Alumni survey": "#87D8F7",  # celeste
                "School survey": "#0b3d91",  # blu scuro
                "Unknown": "#81848b",        # grigio
            }

            for node in G_mi.nodes:
                group = var_type.get(node, "Unknown")
                net_mi.add_node(
                    node,
                    label=node,
                    title=f"{node}<br>Group: {group}",
                    color=NODE_COLORS.get(group, "#bfc7d5")
                )

            # Aggiungi archi con tooltip che mostra il peso
            for u, v, data in G_mi.edges(data=True):
                weight = data.get("weight", 0)
                #tipo = data.get("tipo", "")
                color = "#BDC8D9" #if tipo == "corr" else "gray"

                net_mi.add_edge(u, v, value=weight, title=f"Mutual Information: {weight:.3f}", color=color)

            with col_mi_stat:
                st.caption(f"N. edges: {mi_edges}")

            net_mi.force_atlas_2based()
            html_content_mi = net_mi.generate_html()
            # (inietta JavaScript per centrare e zoomare automaticamente)
            html_content_mi = html_content_mi.replace(
                "</body>",
                """
                <script type="text/javascript">
                    // Attende che la rete sia pronta, poi centra la vista
                    window.addEventListener('load', () => {
                        if (typeof network !== 'undefined') {
                            network.fit();
                        }
                    });
                </script>
                </body>
                """
            )
            st.markdown(
                """
                <br>
                <span style="color:#8fd3ff">●</span> Alumni survey &nbsp; 
                <span style="color:#0b3d91">●</span> School survey &nbsp; 
                """,
                unsafe_allow_html=True
            )
            components.html(html_content_mi, height=750, scrolling=True)

with tab_scenario:
    st.markdown("---")
    st.markdown("## 🧮 Scenario Analysis")
    st.caption("Modify the feature values to simulate alternative scenarios and see the impact on the predicted Rank.")
    if st.session_state.get('shap_df') is None:
        st.warning("To perform Scenario Analysis, you must first train the model in the 'Global Model Analysis (SHAP)' tab.")
    else:
        # Si assume che le librerie necessarie (come pandas, numpy) siano già importate altrove.
        import numpy as np # Necessario per np.array
        import pandas as pd # Aggiunto per sicurezza se non globale
        
        # --- Estrai dati ---
        df = st.session_state['df']
        df_reset = df.reset_index(drop=True)
        feature_cols = st.session_state['model_results']['feature_cols']
        shap_data = st.session_state['shap_data']
        model = shap_data['model']
        X = shap_data['X']

        # --- Selezione università ---
        col_select_uni_scenario, col_empty, col_reset = st.columns([2, 3, 1])
        with col_select_uni_scenario:
            uni_names = sorted(df['School name'].unique())
            luiss_name = "Luiss University/Luiss Business School"
            default_index = list(uni_names).index(luiss_name) if luiss_name in uni_names else 0
            selected_uni_name = st.selectbox("Select University:", uni_names, index=default_index, key='uni_scenario_select')
        uni_idx = df_reset.index[df_reset['School name'] == selected_uni_name][0]
        uni_score_original = float(df_reset.iloc[uni_idx]['Rank'])
        ft_rank_original = int(df_reset['Rank'].max() - uni_score_original + 1)
        # original feature values
        df_features = pd.DataFrame(X, columns=feature_cols)
        original_values = {col: float(df_features.iloc[uni_idx][col]) for col in feature_cols}
        
        # Inizializza o resetta scenario_values quando cambia università
        if 'last_selected_uni' not in st.session_state or selected_uni_name != st.session_state['last_selected_uni']:
            st.session_state['scenario_values'] = original_values.copy()
            st.session_state['last_selected_uni'] = selected_uni_name
            st.session_state['scenario_result'] = None
            # IMPORTANTE: Resetta anche le chiavi degli slider per la nuova università
            for feature in feature_cols:
                if f"slider_{feature}" in st.session_state:
                    del st.session_state[f"slider_{feature}"]
            
        def reset_scenario():
            """Funzione di callback per resettare i valori di scenario e il risultato."""
            st.session_state['scenario_values'] = original_values.copy()
            st.session_state['scenario_result'] = None
            # IMPORTANTE: Resetta anche le chiavi degli slider al valore originale
            for feature in feature_cols:
                #st.session_state[f"slider_{feature}"] = original_values[feature]
                if f"slider_{feature}" in st.session_state:
                    del st.session_state[f"slider_{feature}"]

        with col_reset:
            st.markdown(""); st.markdown("")
            if st.button("🔄 Reset to Original Values", use_container_width=True, on_click=reset_scenario):
                # If reset is called (callback), the state is already updated.
                # Here we do nothing.
                pass
        st.markdown("---")
        
        # --- Layout colonne ---
        col_scores, col_features = st.columns([1, 4])
        
        with col_features:
            st.subheader("⚙️ Modify Feature")
            st.caption("Set the values and press 'Calculate Scenario' to update the score.")

            if 'scenario_result' not in st.session_state:
                st.session_state['scenario_result'] = None
                
            with st.form("scenario_form", clear_on_submit=False):
                col_empty_form, col_submit = st.columns([1, 4])
                with col_submit:
                    submitted = st.form_submit_button("🚀 Calculate Scenario",
                                                       use_container_width=True,
                                                       type='primary')
                
                # LOOP SLIDER dentro la form
                for feature in feature_cols:
                    col_label, col_slider = st.columns([1, 4])
                    
                    # Calcolo min/max/step
                    if feature == 'Weighted salary (US$)':
                        min_val = float(round(float(df_features[feature].min()*0.8) / 100) * 100)
                        max_val = float(round(float(df_features[feature].max()*1.2) / 100) * 100)
                        step = 100.0
                    else:
                        min_val, max_val, step = 0.0, 100.0, 1.0
                        
                    original_val = float(original_values[feature])
                    
                    # Il valore iniziale DEVE essere preso dalla chiave del widget
                    # se è già in sessione, altrimenti da scenario_values
                    if f"slider_{feature}" in st.session_state and selected_uni_name == st.session_state['last_selected_uni']:
                        # Uso il valore memorizzato nella chiave del widget se esiste
                        current_val = st.session_state[f"slider_{feature}"]
                    else:
                        # Altrimenti, usa il valore di scenario (che è l'originale al primo load/cambio università)
                        current_val = float(st.session_state['scenario_values'].get(feature, original_val))

                    
                    with col_label:
                        st.markdown(f"<div style='padding-top:4px;'><strong>{feature}</strong></div>", unsafe_allow_html=True)
                        
                    with col_slider:
                        st.slider(
                            label=feature, #"",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=current_val, # Usa il valore correttamente inizializzato
                            step=float(step),
                            key=f"slider_{feature}",
                            help=f"Original value: {original_val}",
                            label_visibility="collapsed" #"visible"
                        )
                    
            # Gestione submit (fuori dal with st.form)
            if submitted:
                # 1. Aggiorna lo stato dei valori di scenario con i valori finali del form
                for feature in feature_cols:
                    # I valori sono già nelle chiavi degli slider, li usiamo
                    st.session_state['scenario_values'][feature] = st.session_state[f"slider_{feature}"]
                
                # 2. Calcola la previsione
                scenario_X = np.array(
                    [st.session_state['scenario_values'][col] for col in feature_cols],
                    dtype=float
                ).reshape(1, -1)
                predicted_score = float(model.predict(scenario_X)[0])
                predicted_rank = int(df_reset['Rank'].max() - predicted_score + 1)
                score_diff = predicted_score - float(uni_score_original)
                rank_diff = int(ft_rank_original - predicted_rank)
                
                # 3. Salva risultato in sessione
                st.session_state['scenario_result'] = {
                    "predicted_score": predicted_score,
                    "predicted_rank": predicted_rank,
                    "score_diff": score_diff,
                    "rank_diff": rank_diff
                }

        # --- COLONNA SCORE (Nessuna modifica necessaria qui) ---
        with col_scores:
            st.subheader("📊 Results")
            # box punteggio originale
            st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:15px;">
                    <h4 style="margin:0; color:#555;">Original Score</h4>
                    <h2 style="margin:10px 0; color:#333;">{uni_score_original:.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state['scenario_result'] is None:
                st.info("Set the values and press **Calculate Scenario**.")
            else:
                predicted_score = st.session_state['scenario_result']['predicted_score']
                predicted_rank = st.session_state['scenario_result']['predicted_rank']
                score_diff = st.session_state['scenario_result']['score_diff']
                rank_diff = st.session_state['scenario_result']['rank_diff']
                
                invert_rank = st.session_state.get("invert_rank", True)

                if abs(score_diff) < 0.5:
                    box_color, border_color, icon, status = "#e8f4f8", "#b0c4de", "➡️", ""
                elif score_diff > 0 and invert_rank==True:
                    box_color, border_color, icon, status = "#d4f8d4", "#8cd98c", "📈", ""
                elif score_diff < 0 and invert_rank==True:
                    box_color, border_color, icon, status = "#ffe6e6", "#ff9999", "📉", ""

                elif score_diff < 0 and invert_rank==False:
                    box_color, border_color, icon, status = "#d4f8d4", "#8cd98c", "📈", ""
                elif score_diff > 0 and invert_rank==False:
                    box_color, border_color, icon, status = "#ffe6e6", "#ff9999", "📉", ""

                # else:
                #     box_color, border_color, icon, status = "#ffe6e6", "#ff9999", "📉", ""
                    
                st.markdown(f"""
                    <div style="background-color:{box_color}; padding:20px; border-radius:10px; border:2px solid {border_color}; margin-bottom:15px;">
                        <h4 style="margin:0; color:#555;">{icon} Predicted Score</h4>
                        <h2 style="margin:10px 0; color:#333;">{predicted_score:.2f}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="background-color:#fff; padding:15px; border-radius:10px; border:2px solid {border_color};">
                        <h4 style="margin:0 0 10px 0; color:#555;">Variation</h4>
                        <p style="margin:5px 0;"><strong>Score:</strong> {score_diff:+.2f} points</p>
                        <p style="margin:10px 0 0 0; color:#666; font-style:italic;">{status}</p>
                    </div>
                """, unsafe_allow_html=True)