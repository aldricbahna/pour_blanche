import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

COLOR_MAP = {
    'normal': '#90EE90',       # Vert clair (LightGreen)
    'pas_normal': '#F08080'    # Rouge clair (LightCoral)
}

data=pd.read_excel("donnees.xlsx")
data=data.iloc[2:,2:]
data.columns=data.iloc[0,:]
df0=data.iloc[1:,:]

def clean_column(col):
    col = col.strip().lower()
    col = col.replace("'", "")
    col = col.replace("-", " ")
    col = col.replace("(", "").replace(")", "")
    col = col.replace("/", " ")
    col = "_".join(col.split())
    return col
df=df0.copy()
df.columns = [clean_column(c) for c in df0.columns]

df = df.replace(
    {
        "N/A": np.nan,
        "NA": np.nan,
        "n/a": np.nan,
        "na": np.nan,
        "NSP": np.nan,
        "nsp": np.nan,
        "Ne sait pas":np.nan,
        "": np.nan,
        "on":np.nan
    }
)

df=df.rename(columns={'tranche_dâge':'tranche_age',
                      'catégorie_socio_professionnelle':'csp',
                      'nombre_denfant':'nb_enfant',
                      'antécédents_médicaux':'antecedents',
                      'accord_pour_lutilisation_des_données':'accord_donnees',
                      'résultat_du_test':'resultats_test'})
                      
def convert_age(x):
    if x=='> 65 ans':
        return 70
    elif isinstance(x, str) and "-" in x:
        nums = "".join([c if c.isdigit() or c == "-" else " " for c in x]).split("-")
        try:
            a, b = map(int, nums)
            return (a + b) / 2
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

df["tranche_age"] = df["tranche_age"].apply(convert_age)

dict_nb_enfant={'Aucun':0,
                '1 enfant':1,
                '2 enfants':2,
                '3 enfants':3,
                '4 enfants ou plus':4}
df['nb_enfant']=df['nb_enfant'].map(dict_nb_enfant)

df['resultats_test']=df['resultats_test'].replace('non réalisé',np.nan)
df['resultats_test']=df['resultats_test'].replace('absence',np.nan)
df=df.dropna(subset='resultats_test')
      

df['classe_test']=df['resultats_test'].apply(lambda x:'normal' if x=='normal' else 'pas_normal')

st.subheader("Nettoyage du dataset et suppresion des valeurs manquantes pour le résultat du test")
st.dataframe(df)   


df_counts = df['classe_test'].value_counts().reset_index()
df_counts.columns = ['classe', 'count']

fig_pie = px.pie(
    df_counts,
    names='classe',        
    values='count',         
    title='Distribution des 2 classes',
    color='classe',        
    color_discrete_map=COLOR_MAP
)

fig_pie.update_traces(
    textinfo='percent+label',  # Affiche le pourcentage et l'étiquette
    marker=dict(line=dict(color='#000000', width=1)) # Bordure noire
)

st.plotly_chart(fig_pie)


fig1=px.histogram(df,x='tranche_age',color='classe_test',marginal='box',color_discrete_map=COLOR_MAP,title="Répartition des tranches d'âge")
st.plotly_chart(fig1)


feature_cols = [col for col in df.columns if col not in ['classe_test', 'colonne_B']]

for col in feature_cols:

    counts = df.groupby([col, 'classe_test']).size().reset_index(name='count')
    proportions = counts.groupby(col)['count'].transform(lambda x: x / x.sum())
    counts['proportion'] = proportions

    fig = px.bar(
        counts,
        x=col,
        y='proportion',
        color='classe_test',
        barmode='group',
        text='proportion',
        title=f'Distribution des 2 classes pour la variable {col}',
        labels={'proportion': 'Proportion', col: col},
        color_discrete_map=COLOR_MAP,
        height=400
    )

    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis=dict(tickformat=".0%")) # Formate l'axe Y en pourcentage

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Test de Chi-2 entre les variables classe_test et suivi_pathologique")
contingency_table = pd.crosstab(df['classe_test'], df['suivi_pathologique'])
st.dataframe(contingency_table)


# --- 3. Exécution du Test du Chi-2 ---

st.subheader("Résultats du Test du Chi-2 ($\chi^2$)")

chi2, p, dof, expected = chi2_contingency(contingency_table)

# Conversion des résultats en format lisible
df_expected = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

st.markdown("""
| Indicateur | Valeur | Interprétation |
| :--- | :--- | :--- |
| **Statistique $\chi^2$** | `{:.3f}` | Mesure de la différence entre les observations et l'indépendance. |
| **p-value** | `{:.4f}` | Probabilité d'observer ces données si les variables étaient indépendantes. |
| **Degrés de Liberté (ddl)** | `{}` | $(Lignes - 1) \times (Colonnes - 1)$ |
""".format(chi2, p, dof))

st.markdown("---")

# --- 4. Interprétation ---

st.subheader("Interprétation Statistique")

alpha = 0.05 # Seuil de signification conventionnel

if p < alpha:
    st.success(f"""
    **Conclusion : Rejet de l'hypothèse nulle (H0).**
    Avec une p-value de **{p:.4f}** (inférieure à {alpha}), nous concluons qu'il existe une **relation statistiquement significative** entre 'classe\_test' et 'suivi\_pathologique'. Les deux variables ne sont **pas indépendantes**.
    """)
else:
    st.warning(f"""
    **Conclusion : Non-rejet de l'hypothèse nulle (H0).**
    Avec une p-value de **{p:.4f}** (supérieure ou égale à {alpha}), nous n'avons **pas assez de preuves** pour rejeter l'hypothèse que 'classe\_test' et 'suivi\_pathologique' sont **indépendantes**.
    """)

st.markdown("---")


st.subheader("Test de Chi-2 entre les variables classe_test et consommation_tabagique")
contingency_table = pd.crosstab(df['classe_test'], df['consommation_tabagique'])
st.dataframe(contingency_table)

st.subheader("Résultats du Test du Chi-2 ($\chi^2$)")

chi2, p, dof, expected = chi2_contingency(contingency_table)

# Conversion des résultats en format lisible
df_expected = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

st.markdown("""
| Indicateur | Valeur | Interprétation |
| :--- | :--- | :--- |
| **Statistique $\chi^2$** | `{:.3f}` | Mesure de la différence entre les observations et l'indépendance. |
| **p-value** | `{:.4f}` | Probabilité d'observer ces données si les variables étaient indépendantes. |
| **Degrés de Liberté (ddl)** | `{}` | $(Lignes - 1) \times (Colonnes - 1)$ |
""".format(chi2, p, dof))

st.markdown("---")

# --- 4. Interprétation ---

st.subheader("Interprétation Statistique")

alpha = 0.05 # Seuil de signification conventionnel

if p < alpha:
    st.success(f"""
    **Conclusion : Rejet de l'hypothèse nulle (H0).**
    Avec une p-value de **{p:.4f}** (inférieure à {alpha}), nous concluons qu'il existe une **relation statistiquement significative** entre 'classe\_test' et 'suivi\_pathologique'. Les deux variables ne sont **pas indépendantes**.
    """)
else:
    st.warning(f"""
    **Conclusion : Non-rejet de l'hypothèse nulle (H0).**
    Avec une p-value de **{p:.4f}** (supérieure ou égale à {alpha}), nous n'avons **pas assez de preuves** pour rejeter l'hypothèse que 'classe\_test' et 'suivi\_pathologique' sont **indépendantes**.
    """)
   





st.sidebar.markdown("---")