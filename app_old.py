import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import json
import base64
from io import BytesIO
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# --- Configurazione Iniziale di Streamlit ---
st.set_page_config(layout="wide", page_title="Analisi Ranking FT (SHAP)")

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

st.title("Analisi e Interpretabilità del Ranking FT")
st.markdown("*Applicazione Streamlit per l'analisi dei fattori che influenzano il ranking "
            "del Financial Times, utilizzando un modello XGBoost e l'interpretabilità SHAP.*")

# --- Funzioni di Pre-Elaborazione e Modello (Da Notebook Originale) ---

@st.cache_data
def load_data():
    """Carica e pre-processa i dati come nel notebook."""
    path = "../input/FT MIM 2025_elaborato.xlsx"
    df = pd.read_excel(path, sheet_name="MIM 2025 (R)")

    # --- Cleaning ---
    df.drop(columns=['FT_r'], inplace=True)
    df.rename(columns={
        'School Name': 'name',
        'Location, by primary campus': 'location',
        'rank_2025': 'ranking_score'
    }, inplace=True)

    # Inversione dei rank (1 = miglior università)
    for c in df.columns:
        if 'rank' in c:
            df[c] = 100 - df[c] + 1

    # Correzione variabile employed_3m_pct
    df['employed_3m_pct'] = np.select(
        [df['employed_3m_pct'] == 10],
        [100],
        df['employed_3m_pct']
    )
    return df

@st.cache_resource
def train_and_shap(df):
    """Addestra il modello XGBoost e calcola i valori SHAP."""
    random_seed = 3
    
    # Prepara i dati
    feature_cols = [c for c in df.columns if c not in ['name', 'location', 'ranking_score']]
    X = df[feature_cols].values
    y = df['ranking_score'].values
    
    # 1. Training XGBoost con Cross-Validation (per la stima delle performance)
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

    # 2. Train Modello Finale su tutto il dataset (per SHAP)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=random_seed, objective='reg:squarederror'
    )
    xgb_model.fit(X, y)
    y_pred = xgb.XGBRegressor.predict(xgb_model, X)
    
    final_r2 = r2_score(y, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y, y_pred))

    final_results = {
        'R2': final_r2,
        'RMSE': final_rmse
    }

    # 3. SHAP ANALYSIS
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    # Crea un DataFrame per i valori SHAP
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, 'name', df['name'].values)
    shap_df.insert(1, 'location', df['location'].values)
    shap_df.insert(2, 'ranking_score', df['ranking_score'].values)
    shap_df.insert(3, 'predicted_score', y_pred)
    
    # Aggiungi le feature originali a un df SHAP esteso per i plot di dipendenza
    X_with_ranking = np.column_stack([X, y])
    shap_values_extended = np.column_stack([shap_values, np.zeros(len(y))])

    return xgb_model, explainer, shap_values, shap_df, df, feature_cols, X, y, cv_results, final_results, X_with_ranking, shap_values_extended


# --- Funzione di Plotting Universale per Classifiche con Evidenziazione ---
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
        hover_data={'ranking_score': ':.2f', selected_variable: ':.2f'}
    )
    
    fig.update_layout(
        height=800, 
        yaxis={'categoryorder':'array', 'categoryarray': category_order, 'title': None},
        showlegend=False, # Nascondi la legenda Highlight/Default
        margin=dict(l=10, r=10, t=50, b=50)
    )
    return fig


# --- Grafici SHAP Generici (Matplotlib/Plotly) ---

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
        #title="SHAP Feature Importance (Valore SHAP medio assoluto)",
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
    #plt.title('SHAP Summary Plot - Distribuzione degli effetti', fontsize=14, fontweight='bold', pad=20)
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

# --- Grafici SHAP Individuali (Drill-Down) ---

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
    
    # Il rank FT (il "ranking_score" è invertito: 100 è il 1°)
    ft_rank = int(df['ranking_score'].max() - uni_score + 1)
    
    #plt.title(f"{uni_name} - Rank #{ft_rank} (Score: {uni_score:.0f})", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

def plot_shap_force(uni_idx, explainer, shap_values, X, feature_cols):
    """Grafico SHAP Force Plot per una singola università - Matplotlib (Embedding)"""
    
    # Il force plot di SHAP per Matplotlib è complicato da integrare,
    # qui usiamo l'implementazione JS (shap.force_plot) che è più adatta per Streamlit
    # ma la generiamo e la convertiamo in HTML per l'embedding.
    
    # Per ragioni di ambiente, usiamo il `shap.force_plot` standard che genera HTML/JS.
    
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

# --- Funzione per il Focus Competitor (Simulazione output notebook) ---

def get_competitor_analysis(df, shap_df, feature_cols, uni_idx_main, num_competitors=3):
    """
    Simula l'analisi competitor trovando i 3 più simili in termini di valori SHAP
    e generando un report testuale formattato in Markdown.
    """
    
    # Università di riferimento
    uni_name_main = df.iloc[uni_idx_main]['name']
    
    # Calcola la distanza SHAP (Euclidea) tra l'università di riferimento e tutte le altre
    shap_values_main = shap_df.iloc[uni_idx_main][feature_cols].values
    
    distances = {}
    for idx, row in shap_df.iterrows():
        if idx != uni_idx_main:
            shap_values_competitor = row[feature_cols].values
            distance = np.linalg.norm(shap_values_main - shap_values_competitor)
            distances[row['name']] = distance

    # Trova i competitor più vicini (più simili in termini di impatto SHAP)
    sorted_competitors = sorted(distances.items(), key=lambda item: item[1])
    
    # Seleziona i top X
    top_competitors = sorted_competitors[:num_competitors]
    
    # Genera report per l'università principale
    main_uni_data = df.iloc[uni_idx_main]
    main_uni_shap = shap_df.iloc[uni_idx_main][feature_cols]
    
    report = f"### Analisi di Competizione per {uni_name_main} (Score: {main_uni_data['ranking_score']:.0f})\n\n"
    report += "**Obiettivo:** Identificare le università con un 'profilo SHAP' (fattori di impatto sul ranking) più simile.\n\n"
    report += "**Risultati (Top {num_competitors} Competitor per Profilo SHAP):**\n\n"
    
    for rank, (name, distance) in enumerate(top_competitors):
        competitor_data = df.query(f"name == '{name}'").iloc[0]
        competitor_shap = shap_df.query(f"name == '{name}'").iloc[0][feature_cols]
        
        report += f"#### {rank+1}. {name} (Score: {competitor_data['ranking_score']:.0f})\n"
        report += f"* **Similitudine SHAP (Distanza):** {distance:.2f}\n"
        
        # Confronto dei principali fattori SHAP (i primi 3 che spingono di più)
        main_top_factors = main_uni_shap.sort_values(ascending=False).head(3)
        comp_top_factors = competitor_shap.sort_values(ascending=False).head(3)

        report += "* **Fattori più spingenti (SHAP):**\n"
        report += "    * **" + uni_name_main + ":** " + ", ".join(main_top_factors.index) + "\n"
        report += "    * **" + name + ":** " + ", ".join(comp_top_factors.index) + "\n\n"
        
        # Aggiungi un confronto sui valori delle feature chiave
        report += "* **Confronto Valori Chiave (Ranking Score e Weighted Salary):**\n"
        report += f"| Feature | {uni_name_main} (Valore) | {name} (Valore) |\n"
        report += "|---|---|---|\n"
        report += f"| **Ranking Score** | {main_uni_data['ranking_score']:.0f} | {competitor_data['ranking_score']:.0f} |\n"
        report += f"| **Weighted Salary** | {main_uni_data['weighted_salary_usd']:.0f} | {competitor_data['weighted_salary_usd']:.0f} |\n\n"


    return report


# --- Carica e Calcola (Esegue la stima del modello e gli SHAP di default) ---
try:
    with st.spinner("Caricamento...", show_time=True):
        xgb_model, explainer, shap_values, shap_df, df, feature_cols, X, y, cv_results, final_results, X_with_ranking, shap_values_extended = train_and_shap(load_data())
    
    # --- Estrai dati per i plot generici ---
    feature_cols_with_ranking = feature_cols + ['ranking_score']
    
    # Trova gli indici delle 6 feature più importanti per i dependence plots
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': np.abs(shap_values).mean(axis=0)}).sort_values('importance', ascending=False)

except Exception as e:
    st.error(f"Errore durante il caricamento o l'addestramento del modello. Assicurati che il tuo notebook contenga la struttura dati corretta. Errore: {e}")
    st.stop()


# --- Struttura dell'App Streamlit (Tabs) ---
# Suddivisione del Tab 1 in due sotto-tab
tab_esplorazione, tab_modello, tab_drilldown = st.tabs(["Caricamento & Esplorazione Dati", "Analisi Globale Modello (SHAP)", "Drill-Down per Università"])

# -----------------------------------------------------
# TAB 1: Caricamento & Esplorazione Dati
# -----------------------------------------------------
with tab_esplorazione:
    st.header("Tabella di input")
    
    # Visualizzazione Tabella Completa
    st.markdown("Dati caricati e pre-processati (Ranking Score: 100 = Rank #1).")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    
    # Controlli per l'evidenziazione
    col_sel, col_empty = st.columns([1, 3])
    with col_sel:
        # Trova l'indice della LUISS (per preselezionarla)
        luiss_index = list(df['name'].unique()).index("Luiss University/Luiss Business School") if "Luiss University/Luiss Business School" in df['name'].unique() else 0

        selected_unis = st.multiselect(
            "Seleziona Università da evidenziare nei grafici:",
            sorted(df['name'].unique()),
            default=["Luiss University/Luiss Business School"] if "Luiss University/Luiss Business School" in df['name'].unique() else df['name'].unique()[luiss_index:luiss_index+1]
        )

    # Suddivisione dei grafici a barre in due colonne
    col_rank_chart, col_var_chart = st.columns(2)

    with col_rank_chart:
        # Grafico di Classifica Generale (Ranking Score)
        st.subheader("Classifica Generale")
        
        df_rank_plot = df[['name', 'ranking_score']].sort_values('ranking_score', ascending=True).reset_index(drop=True)
        df_rank_plot['FT Rank'] = df_rank_plot.index + 1 # Rank FT (1, 2, 3...)
        
        fig_rank = plot_ranked_variable(
            df_rank_plot,
            'ranking_score',
            selected_unis,
            "Classifica per Ranking Score"
        )
        st.plotly_chart(fig_rank, use_container_width=True)
    
    with col_var_chart:
        # Grafico di Classifica per Variabile Selezionata
        st.subheader("Classifica per Variabile")
        
        numeric_cols = [c for c in df.columns if df[c].dtype != 'object' and c != 'ranking_score' and c != 'name']
        
        selected_variable = st.selectbox(
            "Seleziona la variabile per definire la classifica:",
            numeric_cols,
            index=numeric_cols.index('weighted_salary_usd') if 'weighted_salary_usd' in numeric_cols else 0
        )
        
        fig_var = plot_ranked_variable(
            df,
            selected_variable,
            selected_unis,
            f"Classifica per: '{selected_variable}'"
        )
        st.plotly_chart(fig_var, use_container_width=True)


# -----------------------------------------------------
# TAB 2: Analisi Globale Modello (SHAP)
# -----------------------------------------------------
with tab_modello:
    st.header("Stima del Modello")

    # --- Contenitore 1: Riepilogo Dati e Performance ---
    col_data, col_model = st.columns(2)
    
    with col_data:
        st.subheader("Riepilogo dati di input")
        st.markdown(f"* **Numero di Università:** `{len(df)}`")
        st.markdown(f"* **Numero di Feature:** `{len(feature_cols)}`")
        st.dataframe(df.describe().T.style.format('{:.2f}'), use_container_width=True)
    
    with col_model:
        st.subheader("Performance del Modello XGBoost")
        st.markdown(f"""
        Il modello è stato addestrato per prevedere il **Ranking Score** (dove 100 è il rank #1).
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
        top_n_var = st.slider("Numero di feature da mostrare:", 1, max_features, 9, step=1)
        
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

    # Funzione per creare un file Excel in memoria (simulando l'output del notebook)
    def to_excel_download(df_input, df_importance, df_shap):
        """Crea un file Excel in memoria con i tre sheet e restituisce la stringa base64."""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_input.to_excel(writer, sheet_name='Input Dati', index=False)
            df_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            df_shap.to_excel(writer, sheet_name='SHAP per Università', index=False)
        processed_data = output.getvalue()
        return processed_data

    # Crea il df di importanza per l'export
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': np.abs(shap_values).mean(axis=0)}).sort_values('importance', ascending=False)

    # Genera i dati binari
    excel_data = to_excel_download(df, feature_importance_df, shap_df) 

    # Utilizza st.download_button per un bottone più carino
    # FIX: Rimosso use_container_width=True per un bottone di larghezza automatica
    st.download_button(
        label="⬇️ Scarica file Excel (.xlsx)",
        data=excel_data,
        file_name="analisi_ranking_ft.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------------------------------
# TAB 3: Drill-Down per Università (ex Tab 2)
# -----------------------------------------------------
with tab_drilldown:
    
    
    col_select_uni, col_select_empty = st.columns([1, 5])

    with col_select_uni:
        # Selezione dell'università tramite menu a tendina
        uni_names = sorted(df['name'].unique())
        selected_uni_name = st.selectbox(
            "Cambia l'Università:",
            uni_names,
            index=list(uni_names).index("Luiss University/Luiss Business School") if "Luiss University/Luiss Business School" in uni_names else 0,
            key='uni_drilldown_select' # Aggiunto key per Streamlit
        )
    uni_idx = df.query(f"name == '{selected_uni_name}'").index[0]
    uni_score = df.iloc[uni_idx]['ranking_score']
    ft_rank = int(df['ranking_score'].max() - uni_score + 1)

    st.header(f"Drill-Down per '*{selected_uni_name}*' - Rank #{ft_rank} (Score: {uni_score:.0f})")

    col_waterfall, col_force = st.columns([2, 4])
    with col_waterfall:
        st.subheader(f"Waterfall Plot")
        st.caption("Mostra come i valori di ogni feature spingono la previsione del modello dalla media (`Base Value`) al punteggio previsto (`f(x)`).")
        # Grafico Waterfall
        fig_waterfall = plot_shap_waterfall(uni_idx, explainer, shap_values, X, df, feature_cols)
        st.pyplot(fig_waterfall, use_container_width=True)

    with col_force:
        # Suddivisione verticale (Force Plot e Bullet Points)
        # st.rows() non esiste; usiamo l'approccio classico per creare blocchi verticali
        
        # --- BLOCCO 1: Force Plot (Grafico) ---
        st.subheader("Force Plot")
        st.caption("Le feature in **rosso** aumentano il punteggio previsto (`f(x)`), quelle in **blu** lo diminuiscono.")
        
        # Grafico Force Plot (HTML per interattività)
        shap_html_content = plot_shap_force(uni_idx, explainer, shap_values, X, feature_cols)
        st.components.v1.html(shap_html_content, height=350, scrolling=True) 
        
        # --- BLOCCO 2: Bullet Points (Due Colonne) ---
        st.markdown("---")
        
        # Calcola le feature positive e negative per la lista
        pos_features = [(feature_cols[i], shap_values[uni_idx][i]) for i in range(len(feature_cols)) if shap_values[uni_idx][i] > 0]
        neg_features = [(feature_cols[i], shap_values[uni_idx][i]) for i in range(len(feature_cols)) if shap_values[uni_idx][i] < 0]
        
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

    st.markdown("---")
    st.header("Analisi Competitor per Waterfall Plot e Confronto Dati")
    st.caption("Seleziona i competitor per visualizzare il loro *Waterfall Plot* e confrontare i valori delle loro feature con l'Università di riferimento.")
    
    # --- Controlli per i Competitor ---
    col_filter_loc, col_select_uni_comp = st.columns(2)
    
    # Opzione 1: Filtro per Area Geografica
    locations = ['Tutte le aree geografiche'] + list(sorted(df['location'].unique()))
    with col_filter_loc:
        selected_location = st.selectbox("1. Filtra l'elenco università per Area Geografica:", locations, key='comp_location_filter')
        
    # Applica il filtro geografico prima della multiselezione
    df_filtered = df[df['location'] == selected_location] if selected_location != 'Tutte le aree geografiche' else df
    
    # Rimuovi l'università di riferimento dall'elenco dei competitor
    competitor_uni_names = sorted(df_filtered.query(f"name != '{selected_uni_name}'")['name'].unique())
    
    with col_select_uni_comp:
        # Opzione 2: Multiselezione dei competitor
        selected_competitors = st.multiselect(
            "2. Seleziona uno o più competitor per il confronto:",
            competitor_uni_names,
            key='comp_multiselect'
        )
        
    # --- Report di Analisi Competitor e Plot ---
    if selected_competitors:
        
        # Tabella di Confronto
        st.subheader("Tabella di Confronto (Università di riferimento vs Competitor Selezionati)")
        
        # Colonne chiave da confrontare (aggiungi l'università di riferimento)
        uni_names_to_compare = [selected_uni_name] + selected_competitors
        
        # Filtra i dati per le università selezionate
        df_comparison_temp = df[df['name'].isin(uni_names_to_compare)].drop_duplicates(subset=['name'])
        
        # Imposta l'indice a 'name'
        df_comparison = df_comparison_temp.set_index('name', verify_integrity=True) 
        
        # Colonne iniziali fisse
        initial_cols = ['ranking_score', 'location']
        
        # Lista di TUTTE le feature dal modello (feature_cols)
        all_model_features = feature_cols
        
        # Crea la lista delle feature da visualizzare escludendo quelle fisse per evitare duplicati
        unique_model_features = [col for col in all_model_features if col not in initial_cols]
        
        # Colonne finali uniche per la tabella (che diventeranno l'indice dopo la trasposizione)
        display_cols = initial_cols + unique_model_features

        # Prepara il DataFrame di confronto
        df_comparison_fixed = df_comparison[display_cols].copy()
        
        # Converti le colonne numeriche con la formattazione desiderata
        df_comparison_fixed['ranking_score'] = df_comparison_fixed['ranking_score'].apply(lambda x: f"{x:.0f}")
        df_comparison_fixed['weighted_salary_usd'] = df_comparison_fixed['weighted_salary_usd'].apply(lambda x: f"${x:,.0f}")
        
        # Applica una formattazione generica (2 decimali) al resto delle colonne (le feature)
        numeric_feature_cols = [c for c in unique_model_features if df_comparison_fixed[c].dtype != 'object']
        for col in numeric_feature_cols:
             df_comparison_fixed[col] = df_comparison_fixed[col].apply(lambda x: f"{x:.2f}")

        # Creazione della tabella trasposta per una migliore leggibilità
        df_display = df_comparison_fixed.T
        df_display.columns.name = 'Feature'

        # Styling e formattazione
        styled_table = df_display.style \
            .set_caption(f"Confronto Dati Chiave ({selected_uni_name} vs. Competitor)") \
            .set_properties(**{'border-color': 'lightgray'})
            
        st.dataframe(styled_table, use_container_width=True)
        st.markdown("---")
        
        # 2. Waterfall Plot per ciascun Competitor
        st.subheader("Waterfall Plot SHAP per Competitor")
        st.caption("Ogni grafico mostra la decomposizione del Ranking Score previsto per il competitor selezionato, evidenziando i fattori positivi (rosso) e negativi (blu).")
        
        # Prepara le colonne per i subplot
        n_cols_waterfall = 4 #if len(selected_competitors) > 1 else 1 # Mostra 2 colonne se ci sono almeno 2 competitor
        waterfall_cols = st.columns(n_cols_waterfall)
        
        # Itera sui competitor e genera i plot
        for i, competitor_name in enumerate(uni_names_to_compare):#selected_competitors):
            comp_idx = df.query(f"name == '{competitor_name}'").index[0]
            with waterfall_cols[i % n_cols_waterfall]:
                comp_score = df.iloc[comp_idx]['ranking_score']
                comp_rank = int(df['ranking_score'].max() - comp_score + 1)
                st.markdown(f"**{competitor_name}** (Rank #{comp_rank}, Score: {comp_score:.0f})")
                
                fig_comp_waterfall = plot_shap_waterfall(comp_idx, explainer, shap_values, X, df, feature_cols)
                st.pyplot(fig_comp_waterfall, use_container_width=True)
                
    else:
        st.info("Seleziona almeno un'università competitor per avviare l'analisi di confronto.")