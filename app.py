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

# --- Configurazione Iniziale di Streamlit ---
st.set_page_config(layout="wide", page_title="Analisi Ranking FT (SHAP)")

# --- Inizializzazione Session State ---
# Variabili di stato per i dati caricati e i risultati del modello
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'shap_df' not in st.session_state:
    st.session_state['shap_df'] = None
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

st.title("Analisi del Ranking FT")
st.markdown("*Applicazione per l'analisi dei fattori che influenzano il ranking del Financial Times, utilizzando un modello XGBoost e l'interpretabilità SHAP.*")


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

def load_data_from_upload(uploaded_file, sheet_name):
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
            'School Name': 'name',
            'Location, by primary campus': 'location',
            'rank_2025': 'ranking_score'
        }
        
        # Elimina le colonne non necessarie se presenti
        if 'FT_r' in df.columns:
            df.drop(columns=['FT_r'], inplace=True)
        
        # Rinomina le colonne chiave
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

        # Controlla la presenza delle colonne minime richieste
        required_cols = ['name', 'ranking_score']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Il file Excel deve contenere le colonne: '{' e '.join(col_map.keys())}' o già rinominate in 'name' e 'ranking_score'.")
            return None

        # Inversione dei rank (100 = rank #1)
        for c in df.columns:
            # Controllo più robusto: applica l'inversione solo se 'rank' è nel nome
            # e se il tipo è numerico
            if 'rank' in c and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = 100 - df[c] + 1

        # Correzione variabile employed_3m_pct (se presente)
        if 'employed_3m_pct' in df.columns:
            df['employed_3m_pct'] = np.select(
                [df['employed_3m_pct'] == 10],
                [100],
                df['employed_3m_pct']
            )
            
        # Resetta l'upload_file (necessario per resettare il puntatore e poter rileggere)
        uploaded_file.seek(0)
        
        return df

    except Exception as e:
        st.error(f"Errore durante la pre-elaborazione dei dati: {e}")
        return None

# --- Funzione di Training e SHAP (Memorizzata in session_state) ---

# @st.cache_resource è rimosso per permettere il re-training
def train_and_shap(df):
    """Addestra il modello XGBoost e calcola i valori SHAP. Salva in session_state."""
    start_time = time.time()
    random_seed = 3
    
    # Prepara i dati
    # Assumiamo che tutte le colonne numeriche tranne 'ranking_score' siano feature
    feature_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'ranking_score']
    # Rimuovi anche 'location' che è categorica ma a volte può essere interpretata come numerica se ha valori numerici
    feature_cols = [c for c in feature_cols_all if c not in ['name', 'location']] 
    
    # Rimuovi righe con NaN nelle colonne feature o target
    df_clean = df.dropna(subset=feature_cols + ['ranking_score'])
    
    if len(df_clean) == 0:
        st.error("Nessun dato valido rimasto dopo la rimozione dei NaN nelle colonne feature/target.")
        return None

    X = df_clean[feature_cols].values
    y = df_clean['ranking_score'].values
    
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
    shap_df.insert(0, 'name', df_clean['name'].values)
    shap_df.insert(1, 'location', df_clean['location'].values)
    shap_df.insert(2, 'ranking_score', df_clean['ranking_score'].values)
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


def pretrain_and_shap(df):
    """Carica il modello pre-addestrato di XGBoost e calcola i valori SHAP. Salva in session_state."""
    start_time = time.time()
    random_seed = 3
    
    # Prepara i dati
    # Assumiamo che tutte le colonne numeriche tranne 'ranking_score' siano feature
    feature_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'ranking_score']
    # Rimuovi anche 'location' che è categorica ma a volte può essere interpretata come numerica se ha valori numerici
    feature_cols = [c for c in feature_cols_all if c not in ['name', 'location']] 
    
    # Rimuovi righe con NaN nelle colonne feature o target
    df_clean = df.dropna(subset=feature_cols + ['ranking_score'])
    
    X = df_clean[feature_cols].values
    y = df_clean['ranking_score'].values
    
    # Carica i risultati della CV
    cv_scores_r2 = np.load("./data/cv_scores_r2.npy")
    cv_scores_rmse = np.load("./data/cv_scores_rmse.npy")

    cv_results = {
        'R2_mean': cv_scores_r2.mean(),
        'R2_std': cv_scores_r2.std(),
        'RMSE_mean': cv_scores_rmse.mean(),
        'RMSE_std': cv_scores_rmse.std(),
    }

    # 2. Caricamento Modello Finale e inferenza su tutto il dataset
    xgb_model = joblib.load("./data/xgb_model_final.pkl")
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
    shap_df.insert(0, 'name', df_clean['name'].values)
    shap_df.insert(1, 'location', df_clean['location'].values)
    shap_df.insert(2, 'ranking_score', df_clean['ranking_score'].values)
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
    st.success(f"Modello XGBoost caricato con successo in **{execution_time:.2f} secondi** ⏱️")


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
    df_plot['Color'] = df_plot['name'].apply(lambda x: 'Highlight' if x in highlight_unis else 'Default')

    # Ordine delle categorie per l'asse y
    category_order = df_plot['name'].tolist()

    # Nuovi colori
    color_discrete_map = {
        'Default': "#d9e5f2",
        'Highlight': '#41a4ff'
    }

    # Crea il grafico
    fig = px.bar(
        df_plot,
        x=selected_variable,
        y='name',
        orientation='h',
        title=title,
        labels={selected_variable: selected_variable, 'name': 'Università'},
        color='Color',
        color_discrete_map=color_discrete_map,
        category_orders={'name': category_order},
        hover_data={'ranking_score': ':.0f', selected_variable: ':.2f'} # Aggiornato a .0f per il rank
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
    
    # Utilizza 'ranking_score' come interaction_index per colorare i punti in base al target
    shap.dependence_plot(
        feat_idx, 
        shap_values_extended,  
        X_with_ranking, 
        feature_names=feature_cols_with_ranking,
        interaction_index='ranking_score',  
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
    
    uni_name = df.iloc[uni_idx]['name']
    uni_score = df.iloc[uni_idx]['ranking_score']
    
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
tab_esplorazione, tab_modello, tab_drilldown = st.tabs(["Caricamento & Esplorazione Dati", "Analisi Globale Modello (SHAP)", "Drill-Down per Università"])


# -----------------------------------------------------
# TAB 1: Caricamento & Esplorazione Dati
# -----------------------------------------------------
with tab_esplorazione:
    st.header("Caricamento Dati di Input")
    
    # File Uploader
    col_uploader, col_system_data, col_empty_uploader = st.columns([1, 1, 4])
    with col_uploader:
        uploaded_file = st.file_uploader(
            "Carica il tuo file Excel (.xlsx)", 
            type="xlsx"
        )
        df = st.session_state['df']
        sheet_name = None
        
        if uploaded_file is not None:
            # Se un file è caricato, mostra il selettore di sheet
            sheet_names = get_sheet_names(uploaded_file)
            
            if sheet_names:
                col_sheet, col_empty = st.columns([1, 3])
                with col_sheet:
                    # Trova il foglio che contiene '(R)' come default, se esiste
                    default_index = next((i for i, name in enumerate(sheet_names) if '(R)' in name), 0)
                    
                    sheet_name = st.selectbox(
                        "Seleziona il foglio di lavoro da caricare:",
                        sheet_names,
                        index=default_index,
                        key='sheet_select'
                    )
                    
                # Bottone di caricamento
                if st.button("Upload del file"):
                    # Carica i dati e salva il DF pulito in session_state['df']
                    with st.spinner(f"Caricamento dati dal foglio '{sheet_name}'..."):
                        df_temp = load_data_from_upload(uploaded_file, sheet_name)
                        if df_temp is not None:
                            st.session_state['df'] = df_temp
                            # Resetta i risultati del modello precedente
                            st.session_state['shap_df'] = None
                            st.session_state['model_results'] = None
                            st.session_state['shap_data'] = None
                            st.success("Dati caricati e pronti per l'analisi.")
                        else:
                            st.error("Caricamento fallito. Controlla il formato del file.")
                            st.session_state['df'] = None
                    
                    # Forzare un rerun per aggiornare la visualizzazione
                    st.rerun()
                
                #st.markdown("---")

    with col_system_data:
        st.markdown("Carica file excel già a sistema")
        if st.button("Carica dati predefiniti"):
            example_file_path = "./data/FT MIM 2025.xlsx"
            with open(example_file_path, "rb") as f:
                example_data = f.read()
            st.session_state['df'] = load_data_from_upload(BytesIO(example_data), "MIM 2025 (R)")
            # Resetta i risultati del modello precedente
            st.session_state['shap_df'] = None
            st.session_state['model_results'] = None
            st.session_state['shap_data'] = None
            st.success("Dati di esempio caricati con successo.")
            # Forzare un rerun per aggiornare la visualizzazione
            #st.rerun()
    

    # Contenuto Esplorazione (Mostrato solo se i dati sono stati caricati)
    if st.session_state['df'] is not None:
        st.markdown("---")
        df = st.session_state['df']
        st.header("Tabella di Input")
        
        # Visualizzazione Tabella Completa
        st.markdown(f"Dati caricati e pre-processati (Ranking Score: {df['ranking_score'].max():.0f} = Rank #1).")
        st.markdown(f"**N. di osservazioni:** `{len(df)}` | **N. variabili:** `{len(df.columns)}`")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("---")
        
        # Controlli per l'evidenziazione
        col_sel, col_empty = st.columns([1, 3])
        with col_sel:
            # Trova l'indice della LUISS (per preselezionarla)
            luiss_name = "Luiss University/Luiss Business School"
            default_unis = [luiss_name] if luiss_name in df['name'].unique() else df['name'].unique()[:1].tolist()

            selected_unis = st.multiselect(
                "Seleziona Università da evidenziare nei grafici:",
                sorted(df['name'].unique()),
                default=default_unis,
                key='exp_highlight_unis'
            )

        # Suddivisione dei grafici a barre in due colonne
        col_rank_chart, col_var_chart = st.columns(2)

        with col_rank_chart:
            # Grafico di Classifica Generale (Ranking Score)
            st.subheader("Classifica Generale")
            
            fig_rank = plot_ranked_variable(
                df,
                'ranking_score',
                selected_unis,
                "Classifica per Ranking Score"
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        
        with col_var_chart:
            # Grafico di Classifica per Variabile Selezionata
            st.subheader("Classifica per Variabile")
            
            # Filtra le colonne numeriche disponibili per i plot
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'ranking_score' and c != 'name']
            
            selected_variable = st.selectbox(
                "Seleziona la variabile per definire la classifica:",
                numeric_cols,
                index=numeric_cols.index('weighted_salary_usd') if 'weighted_salary_usd' in numeric_cols else 0,
                key='exp_variable_select'
            )
            
            fig_var = plot_ranked_variable(
                df,
                selected_variable,
                selected_unis,
                f"Classifica per: '{selected_variable}'"
            )
            st.plotly_chart(fig_var, use_container_width=True)
    
    elif uploaded_file is None:
        col_msg, col_msg_empty = st.columns([2, 4])
        with col_msg:
            st.info("Carica dei dati per iniziare.")
    elif st.session_state['df'] is None and uploaded_file is not None and sheet_name is None:
        st.info("Seleziona dati di input e premi 'Carica e Pre-processa Dati'.")


# -----------------------------------------------------
# TAB 2: Analisi Globale Modello (SHAP)
# -----------------------------------------------------
with tab_modello:
    
    st.header("Modello di Machine Learning e Calcolo SHAP")
    
    if st.session_state['df'] is None:
        st.warning("Per addestrare il modello, carica prima i dati nel Tab 'Caricamento & Esplorazione Dati'.")
    else:
        # Bottone per addestrare il modello
        col_bottone1, col_bottone2, col_empty = st.columns([1, 1, 4])
        with col_bottone1:
            if st.button("🚀 Avvia addestramento del modello"):
                with st.spinner("Addestramento in corso... (potrebbe richiedere qualche minuto)"):
                    train_and_shap(st.session_state['df'].copy()) # Passa una copia per evitare side effects
                # L'app si riaggiornerà e visualizzerà i risultati grazie all'if successivo
        with col_bottone2:
            if st.button("♻️ Carica il modello pre-addestrato"):
                with st.spinner("Caricamento del modello..."):
                    pretrain_and_shap(st.session_state['df'].copy()) # Passa una copia per evitare side effects
                # L'app si riaggiornerà e visualizzerà i risultati grazie all'if successivo

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
            
            feature_cols_with_ranking = feature_cols + ['ranking_score']
            
            st.markdown("---")
            st.header("Stima del Modello")

            # --- Contenitore 1: Riepilogo Dati e Performance ---
            col_data, col_model = st.columns(2)
            
            with col_data:
                st.subheader("Riepilogo dati di input utilizzati")
                st.markdown(f"* **Numero di Università:** `{len(df)}`")
                st.markdown(f"* **Numero di Feature:** `{len(feature_cols)}`")
                st.dataframe(df[feature_cols].describe().T.style.format('{:.2f}'), use_container_width=True)
            
            with col_model:
                st.subheader("Performance del Modello XGBoost")
                st.markdown(f"""
                Il modello è stato addestrato per prevedere il **Ranking Score** (dove {df['ranking_score'].max():.0f} è il rank #1).
                * **R² su Training Set:** `{final_results['R2']:.4f}`
                * **RMSE su Training Set:** `{final_results['RMSE']:.4f}`
                * **R² medio (Cross-Validation):** `{cv_results['R2_mean']:.4f} (+/- {cv_results['R2_std']:.4f})`
                * **RMSE medio (Cross-Validation):** `{cv_results['RMSE_mean']:.4f} (+/- {cv_results['RMSE_std']:.4f})`
                """)
                st.info("Nota: L'alto R² sul training set (non CV) suggerisce un buon *fit* per l'analisi SHAP interpretativa.")

            # --- Contenitore 2: Grafici di Importanza e Distribuzione SHAP ---
            st.markdown("---")
            st.header("SHAP Analysis - Interpretabilità del Modello")
            col_bar, col_beeswarm = st.columns(2)

            with col_bar:
                st.subheader("Feature Importance")
                st.caption("Mostra le feature ordinate in modo descrescente per importanza media assoluta nel determinare il Ranking Score.")
                # Grafico a barre con Plotly
                st.plotly_chart(plot_shap_summary_bar(shap_values, feature_cols), use_container_width=True)

            with col_beeswarm:
                st.subheader("SHAP Beeswarm Plot - Distribuzione degli Effetti")
                st.caption("Ogni punto è un'università. Il colore indica il valore della feature, la posizione orizzontale indica l'impatto sul Ranking Score (valore SHAP).")
                # Grafico Beeswarm (Matplotlib, usa la cache)
                st.pyplot(plot_shap_beeswarm(shap_values, X, feature_cols), use_container_width=True)

            # --- Contenitore 3: Dependence Plots (Focus sulle interazioni) ---
            st.markdown("---")
            
            # 1. Slider per la selezione del numero di feature
            col_slider, col_empty = st.columns([3, 7]) # Divisione in colonne per limitare la larghezza dello slider
            
            with col_slider:
                max_features = len(shap_values[0])
                top_n_var = st.slider("Numero di feature da mostrare:", 1, max_features, min(9, max_features), step=1, key='mod_slider_features')
                
            st.subheader(f"SHAP Dependence Plots (Interazioni con il Ranking Score) - Top {top_n_var} Features")
            st.caption("Questi grafici mostrano l'effetto di una singola feature sul Ranking Score, colorando i punti in base al Ranking Score effettivo di quell'università.")
            
            # Trova gli indici delle top_n_var feature più importanti
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-top_n_var:][::-1]

            # Determina la disposizione dei subplot in base al numero di top_n_var
            n_cols = 3
            
            # Crea le colonne per i dependence plots
            all_cols = st.columns(n_cols)
            
            # Itera sui top N indici e genera i plot
            for i, feat_idx in enumerate(top_features_idx):
                with all_cols[i % n_cols]:
                    fig = plot_shap_dependence(feat_idx, shap_values_extended, X_with_ranking, feature_cols_with_ranking)
                    st.pyplot(fig, use_container_width=True)


            # --- Esportazione Dati ---
            st.markdown("---")
            st.subheader("Download dei Dati Elaborati")

            # Crea il df di importanza per l'export
            feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': np.abs(shap_values).mean(axis=0)}).sort_values('importance', ascending=False)

            # Genera i dati binari
            excel_data = to_excel_download(df, feature_importance_df, shap_df) 

            # Utilizza st.download_button per un bottone più carino
            st.download_button(
                label="⬇️ Scarica file Excel (.xlsx)",
                data=excel_data,
                file_name="analisi_ranking_ft_risultati.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Addestra il modello o caricane uno già stimato, e visualizza i risultati SHAP.")


# -----------------------------------------------------
# TAB 3: Drill-Down per Università
# -----------------------------------------------------
with tab_drilldown:
    
    if st.session_state['shap_df'] is None:
        st.warning("Per l'analisi Drill-Down, devi prima addestrare il modello nel Tab 'Analisi Globale Modello (SHAP)'.")
    else:
        # Estrai dati da session state
        df = st.session_state['df']
        # Usiamo una versione con indice resettato per allineare gli indici con X e shap_values
        df_reset = df.reset_index(drop=True) 
        
        shap_df = st.session_state['shap_df']
        feature_cols = st.session_state['model_results']['feature_cols']
        shap_data = st.session_state['shap_data']
        explainer = shap_data['explainer']
        shap_values = shap_data['shap_values']
        X = shap_data['X']
        
        col_select_uni, col_select_empty = st.columns([1, 5])

        with col_select_uni:
            # Selezione dell'università tramite menu a tendina
            uni_names = sorted(df['name'].unique())
            
            luiss_name = "Luiss University/Luiss Business School"
            default_index = list(uni_names).index(luiss_name) if luiss_name in uni_names else 0

            selected_uni_name = st.selectbox(
                "Cambia l'Università:",
                uni_names,
                index=default_index,
                key='uni_drilldown_select' # Aggiunto key per Streamlit
            )
        
        # Estrai i dati per l'università selezionata (per il Drill-Down iniziale)
        # Nota: usiamo l'indice in df_reset per l'accesso a X e shap_values
        uni_idx_in_X_shap = df_reset.query(f"name == '{selected_uni_name}'").index[0] 
        uni_score = df_reset.iloc[uni_idx_in_X_shap]['ranking_score']
        ft_rank = int(df_reset['ranking_score'].max() - uni_score + 1)

        st.header(f"Drill-Down per '*{selected_uni_name}*' - Rank #{ft_rank} (Score: {uni_score:.0f})")

        # ... (Il resto del codice Force Plot e Waterfall Plot del Drill-Down iniziale rimane invariato) ...
        col_waterfall, col_force = st.columns([2, 4])
        with col_waterfall:
            st.subheader(f"Waterfall Plot")
            st.caption("Mostra come i valori di ogni feature spingono la previsione del modello dalla media (`Base Value`) al punteggio previsto (`f(x)`).")
            # Grafico Waterfall
            fig_waterfall = plot_shap_waterfall(uni_idx_in_X_shap, explainer, shap_values, X, df_reset, feature_cols)
            st.pyplot(fig_waterfall, use_container_width=True)

        with col_force:
            # --- BLOCCO 1: Force Plot (Grafico) ---
            st.subheader("Force Plot")
            st.caption("Le feature in **rosso** aumentano il punteggio previsto (`f(x)`), quelle in **blu** lo diminuiscono.")
            
            # Grafico Force Plot (HTML per interattività)
            shap_html_content = plot_shap_force(uni_idx_in_X_shap, explainer, shap_values, X, feature_cols)
            st.components.v1.html(shap_html_content, height=350, scrolling=True) 
            
            # ... (Blocco 2: Feature Positive/Negative) ...
            st.markdown("---")
            
            # Calcola le feature positive e negative per la lista
            pos_features = [(feature_cols[i], shap_values[uni_idx_in_X_shap][i]) for i in range(len(feature_cols)) if shap_values[uni_idx_in_X_shap][i] > 0]
            neg_features = [(feature_cols[i], shap_values[uni_idx_in_X_shap][i]) for i in range(len(feature_cols)) if shap_values[uni_idx_in_X_shap][i] < 0]
            
            pos_features_sorted = sorted(pos_features, key=lambda x: x[1], reverse=True)
            neg_features_sorted = sorted(neg_features, key=lambda x: x[1])

            col_positivi, col_negativi = st.columns(2)
            
            with col_positivi:
                st.markdown("**📈 Top 5 Feature che aumentano il punteggio previsto:**")

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
                st.markdown("**📉 Top 5 Feature che diminuiscono il punteggio previsto:**")

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
        st.header("Analisi Competitor")
        st.caption("Seleziona i competitor per visualizzare il loro *Waterfall Plot* e confrontare i valori delle loro feature con l'Università di riferimento.")
        
        # --- Controlli per i Competitor ---
        col_filter_loc, col_select_uni_comp, col_empty_sel = st.columns([1, 2, 3])
        
        # Opzione 1: Filtro per Area Geografica
        locations = ['Tutte le aree geografiche'] + list(sorted(df['location'].unique()))
        with col_filter_loc:
            selected_location = st.selectbox("Filtra l'elenco università per Area Geografica:", locations, key='comp_location_filter')
            
        # Applica il filtro geografico prima della multiselezione
        df_filtered = df[df['location'] == selected_location] if selected_location != 'Tutte le aree geografiche' else df
        
        # Rimuovi l'università di riferimento dall'elenco dei competitor
        competitor_uni_names = sorted(df_filtered.query(f"name != '{selected_uni_name}'")['name'].unique())
        
        with col_select_uni_comp:
            # Opzione 2: Multiselezione dei competitor
            selected_competitors = st.multiselect(
                "Seleziona uno o più competitor per il confronto:",
                competitor_uni_names,
                key='comp_multiselect'
            )
            
        # --- Report di Analisi Competitor e Plot ---
        if selected_competitors:
            
            # --- 1. Tabella di Confronto (Ordine Forzato) ---
            st.subheader("Tabella di Confronto (Università di riferimento vs Competitor Selezionati)")
            
            # Lista ordinata delle università da confrontare
            uni_names_to_compare = [selected_uni_name] + selected_competitors
            
            # Filtra i dati per le università selezionate
            df_comparison_temp = df[df['name'].isin(uni_names_to_compare)].drop_duplicates(subset=['name'])
            
            # Imposta l'indice a 'name'
            df_comparison = df_comparison_temp.set_index('name', verify_integrity=True) 
            
            # Riordina le colonne in base alla lista ordinata uni_names_to_compare
            df_comparison = df_comparison.reindex(uni_names_to_compare) # <--- CORREZIONE ORDINE TABELLA
            
            # Colonne iniziali fisse e feature
            initial_cols = ['ranking_score', 'location']
            all_model_features = feature_cols
            unique_model_features = [col for col in all_model_features if col not in initial_cols]
            display_cols = initial_cols + unique_model_features

            df_comparison_fixed = df_comparison[display_cols].copy()
            
            # Applica formattazione (assicurati che le colonne esistano prima di formattare)
            if 'ranking_score' in df_comparison_fixed.columns:
                 df_comparison_fixed['ranking_score'] = df_comparison_fixed['ranking_score'].apply(lambda x: f"{x:.0f}")
            if 'weighted_salary_usd' in df_comparison_fixed.columns:
                 df_comparison_fixed['weighted_salary_usd'] = df_comparison_fixed['weighted_salary_usd'].apply(lambda x: f"${x:,.0f}")
            
            numeric_feature_cols = [c for c in unique_model_features if pd.api.types.is_numeric_dtype(df_comparison_fixed[c])]
            for col in numeric_feature_cols:
                 df_comparison_fixed[col] = df_comparison_fixed[col].apply(lambda x: f"{x:.2f}")

            # Creazione della tabella trasposta
            df_display = df_comparison_fixed.T
            df_display.columns.name = 'Feature'

            styled_table = df_display.style \
                .set_caption(f"Confronto Dati Chiave ({selected_uni_name} vs. Competitor)") \
                .set_properties(**{'border-color': 'lightgray'})
                
            st.dataframe(styled_table, use_container_width=True)
            st.markdown("---")
            
            # --- 2. Waterfall Plot per ciascun Competitor (Ordine e Indicizzazione Corretti) ---
            st.subheader("Waterfall Plot SHAP per Competitor")
            st.caption("Ogni grafico mostra la decomposizione del Ranking Score previsto. L'ordine dei grafici riflette l'ordine delle colonne nella tabella.")
            
            n_cols_waterfall = 4 
            # Qui usiamo un numero fisso di 4 colonne, Streamlit gestirà il wrapping se ci sono meno di 4 plot.
            waterfall_cols = st.columns(n_cols_waterfall)
            
            # Itera sulla lista ORDINATA (uni_names_to_compare)
            for i, competitor_name in enumerate(uni_names_to_compare):
                
                # TROVA L'INDICE CORRETTO IN X/SHAP_VALUES usando df_reset
                try:
                    comp_idx_in_X_shap = df_reset.query(f"name == '{competitor_name}'").index[0]
                except IndexError:
                    # Gestisce il caso (improbabile se i dati sono coerenti) in cui il nome non si trovi in df_reset
                    st.error(f"Errore: L'università {competitor_name} non è stata trovata nel dataset di training pulito.")
                    continue
                
                # Usa solo le colonne disponibili (max 4)
                with waterfall_cols[i % n_cols_waterfall]:
                    comp_score = df_reset.iloc[comp_idx_in_X_shap]['ranking_score']
                    comp_rank = int(df_reset['ranking_score'].max() - comp_score + 1)
                    st.markdown(f"**{competitor_name}** (Rank #{comp_rank}, Score: {comp_score:.0f})")
                    
                    # Usa comp_idx_in_X_shap per accedere a X e shap_values
                    # e df_reset per accedere alle info generali
                    fig_comp_waterfall = plot_shap_waterfall(comp_idx_in_X_shap, explainer, shap_values, X, df_reset, feature_cols)
                    st.pyplot(fig_comp_waterfall, use_container_width=True)
                    
        else:
            st.info("Seleziona almeno un'università competitor per avviare l'analisi di confronto.")