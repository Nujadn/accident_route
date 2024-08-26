
## Etape 4 : Déploiement de l'application avec streamlit

# import des librairies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# librairie pour afficher toutes les colonnes du dataframe
pd.set_option('display.max_columns', None)
pd.options.display.max_info_columns

#  décorateur @st.cache_data permet de garder en cache les données chargées, même quand l’application, donc le code, est mise à jour
@st.cache_data 
def load_data1():
   return pd.read_csv("dataset/dataset_analyse.csv")

@st.cache_data 
def load_data2():
   return pd.read_csv("dataset/dataset_to_prepro.csv")
 
@st.cache_data 
def load_data3():
   return pd.read_csv("dataset/dataset_final.csv") 
 
df_ana = load_data1()
df_prepro = load_data2()
df_model = load_data3()


st.title("Projet de prédiction de la gravité des accidents")
st.sidebar.title("Sommaire")
pages=["Contexte", "Le jeu de données", "Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

# contexte
if page == pages[0] : 
  st.write("### Introduction")
  st.write('Le projet « Accidents routiers en France » s’inscrit dans le cadre du cursus Datascientist, formation continue proposée par l’école Datascientest.')
  st.write("L’objectif de ce projet est de prédire la gravité des accidents routiers en France. Les prédictions seront basées sur les données historiques, à partir des données disponibles sur [data.gouv.fr/](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/).")
  st.write("La première étape est d’étudier et appliquer des méthodes pour nettoyer le jeu de données. La deuxième étape est de créer un modèle prédictif. Une fois l’entraînement du modèle effectué, nous allons comparer notre modèle avec les données. ")

# Le jeu de données 
if page == pages[1] : 
  st.write("### Introduction")
  
  
# Exploration  
if page == pages[2] : 
  st.write("### Introduction")  
  st.dataframe(df_ana.head(10))
  st.write(df_ana.shape)
  st.dataframe(df_ana.describe())
  if st.checkbox("Afficher les NA") :
    st.dataframe(df_ana.isna().sum())
    
  fig = plt.figure(figsize=(5,5))
  plt.title("Répartition des types de données")
  df_ana.dtypes.value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral'])
  plt.ylabel('')
  st.pyplot(fig)
  

  
# Dataviz  
if page == pages[3] : 
  st.write("### Introduction") 
  
  df_temp=df_prepro.copy()
  df_temp["lat"] = [value.replace(',', '.') for value in df_temp["lat"]]
  df_temp["lat"] = df_temp["lat"].astype(float)
  df_temp["long"] = [value.replace(',', '.') for value in df_temp["long"]]
  df_temp["long"] = df_temp["long"].astype(float)
  
  df_temp = df_temp.rename(columns={'lat': 'latitude', 'long': 'longitude'})
  lat_min, lat_max = 41.303, 51.124
  long_min, long_max = -5.142, 9.562
  df_france = df_temp[
    (df_temp['latitude'] >= lat_min) &
    (df_temp['latitude'] <= lat_max) &
    (df_temp['longitude'] >= long_min) &
    (df_temp['longitude'] <= long_max)
]
  df_france['an'] = pd.to_numeric(df_france['an'], errors='coerce')
  df_france['longitude'] = pd.to_numeric(df_france['longitude'], errors='coerce')
  df_france['latitude'] = pd.to_numeric(df_france['latitude'], errors='coerce')
  selected_year = st.slider('Choisissez une année:', min_value=int(df_france['an'].min()), max_value=int(df_france['an'].max()), value=int(df_france['an'].max()))
  selected_grav = st.selectbox('Choisissez la gravité de l\'accident:', options=df_france['grav'].unique())

  df_filtered = df_france[(df_france['an'] == selected_year) & (df_france['grav'] == selected_grav)]

  st.map(df_filtered, size=20)
   
  # distribution des accidents par genre 
  fig = plt.figure(figsize=(5,5))
  df_modified = df_ana.copy()
  df_modified['sexe'] = df_modified['sexe'].replace({
   -1: 'non renseigné',
    1: 'Homme',
    2: 'Femme',   
   })
  sns.countplot(x='sexe', data=df_modified)
  plt.xticks(rotation=45, ha='right')
  plt.title('Répartition du nombre des accidents par sexe de l´usager')
  plt.tight_layout()  
  st.pyplot(fig)
  
  
  # Représentation de la gravité des blessures des usagers
  fig = plt.figure(figsize=(5,5))
  #sns.countplot(x = df_ana['grav'])
  grav = ['non renseigné ', 'indemne', 'Tué', 'Blessé hospitalisé', 'Blessé léger'] 
  ax = sns.countplot(x=df_ana['grav'], palette="muted")
  total = len(df_ana['grav'])
  for p in ax.patches:
      height = p.get_height()
      percentage = f'{100 * height / total:.1f}%'
      ax.text(p.get_x() + p.get_width() / 2., height / 2, 
      percentage, ha="center", va="center", color="black", fontsize=12)
  plt.xticks(range(0, len(df_ana['grav'].value_counts())), grav, rotation = 55)  
  plt.title("Représentation de la gravité des blessures des usagers" )
  plt.xlabel("Gravité")
  plt.ylabel("Nombre d'usagers")
  st.pyplot(fig)
  
  # création d'un df dont le genre de la personne est renseigné
  fig, ax = plt.subplots(figsize=(10, 6))
  df_genre = df_ana.copy()
  df_genre = df_ana.loc[(df_ana['sexe'] == 1) | (df_ana['sexe'] == 2)]
  df_genre['sexe'] = pd.to_numeric(df_genre['sexe'], errors='coerce')
  colors = {1: 'blue', 2: 'red'}  
  labels = {1: 'Homme', 2: 'Femme'}
  selected_genre = st.selectbox('Choisissez le genre:', options=df_genre['sexe'].unique())
  df_tempb = df_genre[df_genre['sexe'] == selected_genre]   
  grav_counts = df_tempb['grav'].value_counts().sort_index() 
  fig, ax = plt.subplots(figsize=(10, 6))
  grav_counts = df_tempb['grav'].value_counts().sort_index()
  ax.bar(grav_counts.index, grav_counts.values, color=colors[selected_genre], alpha=0.7)
  ax.set_xlabel('Gravité')
  ax.set_ylabel('Nombre d\'accidents')
  ax.set_title(f'Représentation de la gravité des accidents pour le genre: {labels[selected_genre]}')
  ax.set_xticks([1, 2, 3, 4])
  ax.set_xticklabels(['Indemne', 'Tué', 'Blessé hospitalisé', 'Blessé léger'], rotation=30)
  st.pyplot(fig)
  
  fig, ax = plt.subplots(figsize=(10, 6))
  df_col = df_ana.loc[(df_ana['col'] >= 1) & (df_ana['grav'] >= 1)]
  collision_labels = {
    1: 'Collision avec véhicule',
    2: 'Collision avec piéton',
    3: 'Collision avec animal',
    4: 'Sortie de route',
    5: 'Collision avec obstacle',
    6: 'Collision multiple',
    7: 'Autres'
  }
  cross_tab = pd.crosstab(df_col['col'], df_col['grav'], rownames=['Type de Collision'], colnames=['Gravité'])
  pastel_palette = sns.color_palette("pastel", n_colors=cross_tab.shape[1])
  for i, col in enumerate(cross_tab.columns):
      plt.bar(cross_tab.index, cross_tab[col], bottom=cross_tab.iloc[:, :i].sum(axis=1), color=pastel_palette[i], label=f'Gravité {col}')
  plt.xlabel('Type de collision')
  plt.ylabel('Nombre d\'accidents')
  plt.title('Nombre d\'accidents par type de collision et en fonction de la gravité')
  plt.xticks([1,2,3,4,5,6,7], ["Deux véhicules - frontale","Deux véhicules – par l’arrière","Deux véhicules – par le coté",
                            "Trois véhicules et plus – en chaîne","Trois véhicules et plus - collisions multiples","Autre collision","Sans collision"], rotation=45)
  plt.legend(title='Gravité', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()
  st.pyplot(fig)
  
  fig, ax = plt.subplots(figsize=(5, 5))
  df_annee = df_ana.loc[df_ana['grav'] >= 1]
 # plt.plot(df_annee['an'], df_annee['grav'])
  graph = df_annee.groupby(['an', 'grav']).size().unstack()
  graph.plot(ax=ax)
  plt.title("Accidents entre 2019 à 2022")
  plt.xlabel("Année d'accident")
  plt.ylabel("Nombre d'individu")
  plt.xticks([2019, 2020, 2021, 2022], rotation=30)
  st.pyplot(fig)
  
  
  fig = plt.figure(figsize=(12,8))
  df_ana['date_obj'] = pd.to_datetime(df_ana['an'].astype(str) + '-' + df_ana['mois'].astype(str) + '-' + df_ana['jour'].astype(str))
  df_ana['weekday'] = df_ana['date_obj'].apply(lambda x: x.weekday())
  df_mod = df_ana.copy()
  df_mod['weekday'] = df_mod['weekday'].replace({
   0: 'Lundi',
   1: 'Mardi',
   2: 'Mercredi',
   3: 'Jeudi',
   4: 'Vendredi',
   5: 'Samedi',
   6: 'Dimanche'
  })
  sns.countplot(x='weekday', data=df_mod)
  plt.xticks(rotation=45, ha='right')  # Rotation de 45 degrés, alignées à droite
  plt.title('Répartition des accidents en fonction des jours de la semaine')
  plt.tight_layout()  # Ajuster la mise en page pour éviter la coupure des étiquettes
  st.pyplot(fig)
  
  
  fig = plt.figure(figsize=(12,8))
  vma_t =[30,50,70, 80, 90, 110, 130]
  df_lieux = df_ana[df_ana['vma'].isin(vma_t)].copy()
  sns.boxplot(x = 'vma', data = df_lieux) 
  plt.xlabel('vitesse maximale')
  plt.title("Vitesse maximale autorisée sur le lieu de l'accident")
  st.pyplot(fig)
  
  
  # mode de déplacement
  fig = plt.figure(figsize=(12,8))
  df_ana['catv'] = df_ana['catv'].mask(df_ana['catv'] < 0, 0)
  df_ana['catv'] = df_ana['catv'].replace([50, 60, 41, 42, 43], 100)
  df_ana['catv'] = df_ana['catv'].replace([1, 80], 200)
  df_ana['catv'] = df_ana['catv'].replace([2, 4, 5, 6, 30, 31, 32, 33, 34, 35, 36], 300)
  df_ana['catv'] = df_ana['catv'].replace([18, 19, 37, 38, 39, 40], 400)
  df_ana['catv'] = df_ana['catv'].replace([10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 99], 500)
  df_ana['catv'] = df_ana['catv'].replace([3, 7, 8, 9], 600)
  df_ana['catv'] = df_ana['catv'].replace([100, 200, 300, 400, 500, 600], [1, 2, 3, 4, 5, 6])
  df_ana.catv.value_counts()
  plt.hist(df_ana['catv'],  color = ['#f27750'], label = ['catv'])
  plt.xlabel("Mode de déplacement")
  plt.ylabel("Nombre d'accidents")
  plt.xticks([0,1, 2, 3,4,5,6], ['Marche', 'EDPM', "Vélo", "2RM", "VT", "VU", "Voiture"])
  plt.title('Distribution des accidents par mode de déplacement')
  st.pyplot(fig)
  
# Modélisation  
if page == pages[4] : 
  st.write("### Introduction")  
