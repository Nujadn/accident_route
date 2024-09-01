
## Etape 4 : Déploiement de l'application avec streamlit

# import des librairies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report, f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

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

pages=["Contexte", "Le jeu de données", "Analyse des données", "Préprocessing", "Prédictions", "Conclusion & Perspectives"]

page=st.sidebar.radio("Aller vers", pages)

########################################### contexte #################################################   

if page == pages[0] : 
  st.title("Introduction")
  st.write('Le projet « Accidents routiers en France » s’inscrit dans le cadre du cursus Datascientist, formation continue proposée par l’école Datascientest.')
  st.markdown("""
     Dans le cadre de la Décennie d’action pour la sécurité routière, les Nations Unis ont fixé pour objectif de réduire d’au moins 50 % le nombre d’accidents mortels. Les accidents de la route constituent un problème majeur de santé publique, entraînant de graves conséquences pour les victimes et leurs familles. En plus des blessures physiques, ils peuvent bouleverser la vie des victimes en les privant de leur autonomie, parfois de manière temporaire ou permanente. Les familles sont souvent amenées à réorganiser leur quotidien pour s'occuper des blessés, ce qui peut également perturber leur vie personnelle et professionnelle. Ce qui rajoute un impact économique pour les victimes, leurs familles et leur pays.         
              """)
  st.write("-------------")
    
  st.write("#### Objectif")
  st.write("L’objectif de ce projet est de prédire la gravité des accidents routiers en France. Les prédictions seront basées sur les données historiques, à partir des données disponibles sur [data.gouv.fr/](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/).")


  
  st.write("La première étape a été d’étudier et appliquer des méthodes pour nettoyer le jeu de données. La deuxième étape a été de créer un modèle prédictif. Une fois l’entraînement du modèle effectué, nous avons comparer notre modèle avec les données. ")

########################################### Le jeu de données #################################################   
 
if page == pages[1] : 
  st.title("Introduction")
  st.markdown("""
     Un accident corporel (mortel et non mortel) de la circulation routière relevé par les forces de l’ordre : 
     - implique au moins une victime, 
     - survient sur une voie publique ou privée, ouverte à la circulation publique, 
     - implique au moins un véhicule.
     """)
  st.write("-------------")
  
  st.write("#### MCD des données")
  st.write("Afin de pouvoir prédire la gravité des accidents de la route en France, nous nous sommes principalement basés sur les datasets fournis par le [Ministère de l’intérieur et des Outre-Mer](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/). Les bases de données sont réparties annuellement en 4 fichiers au format csv, de 2005 à 2022 :")
  st.write("Chaque accident corporel où une unité des forces de l’ordre (police, gendarme, etc) est intervenu, a été renseigné dans une fiche intitulée “bulletin d’analyse des accidents corporels”, aussi appelé “fichier BAAC”. Chaque ligne saisie détaille, de manière anonymisée, le lieu et les caractéristiques de l’accident, les véhicules impliqués et leurs victimes. ")
  st.image(Image.open('images/acc_route.jpg'), caption='MCD des données', use_column_width=True)
  
  st.write("-------------")
  st.write("##### La table caractéristiques :")
  st.write("La table décrit les circonstances générales de l’accident. Le numéro d'identifiant de l’accident présent dans ces 4 rubriques permet d'établir un lien entre toutes les variables qui décrivent un accident. ")
  st.write("##### La table lieux :")
  st.write("La table décrit le lieu principal de l’accident même si celui-ci s’est déroulé à une intersection.")
  st.write("##### La table usagers :")
  st.write("La table décrit le profil du ou des usagers impliqué(s) dans l'accident. Ces informations ne divulguent pas les données spécifiques des usagers et des véhicules ainsi que leur comportement qui pourraient porter atteinte à leur vie privée et qui pourraient les rendre identifiables, ou qui pourraient leur porter préjudice.")
  st.write("##### La table véhicules :")
  st.write("La table décrit le ou les véhicules impliqué(s) dans l'accident. Quand un accident comporte plusieurs véhicules, il faut aussi pouvoir le relier chaque véhicule à ses occupants. Ce lien est fait par l'identifiant du véhicule. ")

  st.write("#### Variable cible")
  st.markdown("""
      Un accident corporel implique un certain nombre d’usagers. Parmi ceux-ci, on distingue :
      - les personnes indemnes : impliquées non décédées et dont l’état ne nécessite aucun soin médical du fait de l’accident, les victimes : impliquées non indemnes.
      - les personnes tuées : personnes qui décèdent du fait de l’accident, sur le coup ou dans les trente jours qui suivent l’accident, les personnes blessées : victimes non tuées.
      - les blessés dits « hospitalisés » : victimes hospitalisées plus de 24 heures,
      - les blessés légers : victimes ayant fait l'objet de soins médicaux mais n'ayant pas été admises comme patients à l'hôpital plus de 24 heures.
              """)
  if st.checkbox("Afficher le dataframe") :
    st.dataframe(df_ana.head())
  

    ########################################### Analyse des données = Dataviz#################################################   

if page == pages[2] : 
  st.markdown("<h2 style='text-align: center;'>Données Clés</h2>", unsafe_allow_html=True)
  
  col1, col2 = st.columns(2)
  with col1:
    st.write("#### Forme du DataFrame")
    st.write(f"Nombre de lignes : {df_ana.shape[0]}")
    st.write(f"Nombre de colonnes : {df_ana.shape[1]}")
    st.write(f"Valeurs manquantes : {df_ana.isna().sum().sum()}")
  with col2:  
    st.write("#### Informations : ")
    st.write(f"Nombre d'accidents : {df_ana.Num_Acc.nunique()}" )
    st.write(f"Nombre d'usagers : {df_ana.id_usager.nunique()}" )
    st.write(f"Nombre de véhicules : {df_ana.id_vehicule.nunique()}" )
  st.write("-------------")
    
  st.markdown("<h2 style='text-align: center;'>Répartition des types de données</h2>", unsafe_allow_html=True)
  fig = plt.figure(figsize=(3,3))
  df_ana.dtypes.value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral'])
  plt.ylabel('')
  st.pyplot(fig)  
  st.write("-------------")
  
  st.markdown("<h2 style='text-align: center;'>Représentation géographique des accidents de la route</h2>", unsafe_allow_html=True)
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
  replacement_dict = {
    -1: 'Non renseigné',
    1: 'Indemne',
    2: 'Décédé',
    3: 'Blessé hospitalisé',
    4: 'Blessé léger'
}
  df_france["grav"] = df_france["grav"].replace(replacement_dict)
  selected_year = st.slider('Choisissez une année:', min_value=int(df_france['an'].min()), max_value=int(df_france['an'].max()), value=int(df_france['an'].max()))
  selected_grav = st.selectbox('Choisissez la gravité de l\'accident:', options=df_france["grav"].unique())

  df_filtered = df_france[(df_france['an'] == selected_year) & (df_france['grav'] == selected_grav)]

  st.map(df_filtered, size=20)
  st.write("-------------")
   
  # distribution des accidents par genre 
  st.markdown("<h2 style='text-align: center;'>Répartition du nombre des accidents par sexe de l´usager</h2>", unsafe_allow_html=True)  
  fig = plt.figure(figsize=(5,5))
  df_modified = df_ana.copy()
  df_modified['sexe'] = df_modified['sexe'].replace({
   -1: 'non renseigné',
    1: 'Homme',
    2: 'Femme',   
   })
  sns.countplot(x='sexe', data=df_modified)
  plt.xticks(rotation=45, ha='right')
#  plt.title('Répartition du nombre des accidents par sexe de l´usager')
  plt.tight_layout()  
  st.pyplot(fig)
  st.write("-------------")
  
  # Représentation de la gravité des blessures des usagers
  st.markdown("<h2 style='text-align: center;'>Représentation de la gravité des blessures des usagers</h2>", unsafe_allow_html=True)  
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
#  plt.title("Représentation de la gravité des blessures des usagers" )
  plt.xlabel("Gravité")
  plt.ylabel("Nombre d'usagers")
  st.pyplot(fig)
  st.write("-------------")
  
  
  # création d'un df dont le genre de la personne est renseigné
  st.markdown("<h2 style='text-align: center;'>Représentation de la gravité des accidents en fonction du genre</h2>", unsafe_allow_html=True)  
  fig, ax = plt.subplots(figsize=(10, 6))
  df_genre = df_ana.copy()
  df_genre = df_ana.loc[(df_ana['sexe'] == 1) | (df_ana['sexe'] == 2)]
  df_genre['sexe'] = pd.to_numeric(df_genre['sexe'], errors='coerce')
  colors = {1: 'blue', 2: 'red'}  
  labels = {1: 'Homme', 2: 'Femme'}
  df_genre["sexe"] = df_genre["sexe"].replace(labels)
  selected_genre = st.selectbox('Choisissez le genre:', options=df_genre['sexe'].unique())
  df_tempb = df_genre[df_genre['sexe'] == selected_genre]   
  grav_counts = df_tempb['grav'].value_counts().sort_index() 
  
  ax.bar(grav_counts.index, grav_counts.values, color=colors[1 if selected_genre == 'Homme' else 2], alpha=0.7)
  ax.set_xlabel('Gravité')
  ax.set_ylabel('Nombre d\'accidents')
#  ax.set_title(f'{labels[selected_genre]}')
  ax.set_xticks([1, 2, 3, 4])
  ax.set_xticklabels(['Indemne', 'Tué', 'Blessé hospitalisé', 'Blessé léger'], rotation=30)
  st.pyplot(fig)
  st.write("-------------")
  
  # graph gravité par type de collision
  st.markdown("<h2 style='text-align: center;'>Nombre d'accidents par type de collision et en fonction de la gravité</h2>", unsafe_allow_html=True)    
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
  df_col["grav"] = df_col["grav"].replace(replacement_dict)
  cross_tab = pd.crosstab(df_col['col'], df_col['grav'], rownames=['Type de Collision'], colnames=['Gravité'])
  pastel_palette = sns.color_palette("pastel", n_colors=cross_tab.shape[1])
  for i, col in enumerate(cross_tab.columns):
      plt.bar(cross_tab.index, cross_tab[col], bottom=cross_tab.iloc[:, :i].sum(axis=1), color=pastel_palette[i], label=f'{col}')
  plt.xlabel('Type de collision')
  plt.ylabel('Nombre d\'accidents')
#  plt.title('Nombre d\'accidents par type de collision et en fonction de la gravité')
  plt.xticks([1,2,3,4,5,6,7], ["Deux véhicules - frontale","Deux véhicules – par l’arrière","Deux véhicules – par le coté",
                            "Trois véhicules et plus – en chaîne","Trois véhicules et plus - collisions multiples","Autre collision","Sans collision"], rotation=45)
  plt.legend(title='Gravité', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()
  st.pyplot(fig)
  st.write("-------------")
  
# graph Accidents entre 2019 à 2022
  st.markdown("<h2 style='text-align: center;'>Nombre d'accidents entre 2019 à 2022</h2>", unsafe_allow_html=True)     
  fig, ax = plt.subplots(figsize=(10,6))
  df_annee = df_ana.loc[df_ana['grav'] >= 1]
  df_annee['grav'] = df_annee['grav'].replace(replacement_dict)
 # plt.plot(df_annee['an'], df_annee['grav'])
  graph = df_annee.groupby(['an', 'grav']).size().unstack()
  graph.plot(ax=ax)
#  plt.title("Accidents entre 2019 à 2022")
  plt.xlabel("Année d'accident")
  plt.ylabel("Nombre d'individu")
  plt.xticks([2019, 2020, 2021, 2022], rotation=30)
  ax.legend(bbox_to_anchor=(1, 1))
  plt.tight_layout()
  st.pyplot(fig)
  st.write("-------------")
  
  # graph 
  st.markdown("<h2 style='text-align: center;'>Répartition des accidents en fonction des jours de la semaine</h2>", unsafe_allow_html=True)     
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
  plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Lundi', 'Mardi', "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],rotation=45, ha='right')  # Rotation de 45 degrés, alignées à droite
#  plt.title('Répartition des accidents en fonction des jours de la semaine')
  plt.tight_layout()  # Ajuster la mise en page pour éviter la coupure des étiquettes
  st.pyplot(fig)
  st.write("-------------")
  
 # graph sur la vitesse
  st.markdown("<h2 style='text-align: center;'>Vitesse maximale autorisée sur le lieu de l'accident</h2>", unsafe_allow_html=True)      
  fig = plt.figure(figsize=(12,8))
  vma_t =[30,50,70, 80, 90, 110, 130]
  df_lieux = df_ana[df_ana['vma'].isin(vma_t)].copy()
  sns.boxplot(x = 'vma', data = df_lieux) 
  plt.xlabel('vitesse maximale')
#  plt.title("Vitesse maximale autorisée sur le lieu de l'accident")
  plt.tight_layout()
  st.pyplot(fig)
  st.write("-------------")
  
  # mode de déplacement
  st.markdown("<h2 style='text-align: center;'>Distribution des accidents par mode de déplacement</h2>", unsafe_allow_html=True)      
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
#  plt.title('Distribution des accidents par mode de déplacement')
  st.pyplot(fig)
  st.write("-------------")
  

    ############################################### Préprocessing ###################################################   
 
if page == pages[3] : 
  
  def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
  
   
  st.write("#### Nettoyage des données")
  if st.checkbox("Suppression des doublons") :
    st.write(df_ana.duplicated(keep=False).sum() )
  if st.checkbox("Afficher les valeurs manquantes") :
    st.dataframe(missing_values_table(df_ana) )
  st.write("-------------")
 
  st.write("#### Transformation des variables")  
  
  st.write("##### Nombre de voies") 
  df_avprepro_nbv = df_ana.copy()
  df_apprepro_nbv = df_model.copy()
  option = st.selectbox(
    "Choisissez la distribution :",
    ('Avant le préprocessing', 'Après le préprocessing')
      )
  if option == 'Avant le préprocessing':
    plt.figure(figsize=(8, 4))
    sns.countplot(x='nbv', data=df_avprepro_nbv, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Nombre de voies")
    st.pyplot(plt)
  elif option == 'Après le préprocessing':
    plt.figure(figsize=(8, 4))
    sns.countplot(x='nbv', data=df_apprepro_nbv, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Nombre de voies")
    st.pyplot(plt)
  st.write("-------------")
  
   
  st.write("##### Département et commune")
  col1, col2 = st.columns(2)
  with col1 :
    df_dep = df_ana.loc[(df_ana['dep'] == '2A') | (df_ana['dep'] == '2B')]
    st.write(df_dep['dep'].value_counts())
  with col2 : 
    df_dep['dep'] = df_dep['dep'].str.replace('2A', '20')
    df_dep['dep'] = df_dep['dep'].str.replace('2B', '20')
    st.write(df_dep['dep'].value_counts())
  st.write("-------------") 
    
  st.write("####  Création de nouvelles variables")
  st.write("##### Création d'une tranche horaire") 
  df_avprepro_h = df_ana.copy()
  sepa_values = [value.split(':') for value in df_avprepro_h["hrmn"] ]
  df_avprepro_h["heure"] = [value[0] for value in sepa_values]
  df_apprepro_h = df_model.copy()
  option = st.selectbox(
    "Choisissez la distribution :",
    ("Avant le préprocessing de l'heure", "Après le préprocessing de l'heure")
      )
  if option == "Avant le préprocessing de l'heure":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='heure', data=df_avprepro_h, palette='pastel', order=sorted(df_avprepro_h['heure'].unique()))
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Heure")
    st.pyplot(plt)
  elif option == "Après le préprocessing de l'heure":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='h_group', data=df_apprepro_h, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Heure")
    plt.xticks([0,1, 2, 3,4,5,6, 7], ['1 à 5H', '5 à 7H', "7 à 10H", "10 à 13H", "13 à 16H", "16 à 19H", "19 à 22H", "22 à 1H"])
    st.pyplot(plt)
  st.write("-------------")
  
  st.write("##### Création d'une tranche d'âge") 
  df_avprepro_age = df_ana.copy()
  df_avprepro_age['age'] = df_avprepro_age['an'] - df_avprepro_age['an_nais']
  df_apprepro_age = df_model.copy()
  option = st.selectbox(
    "Choisissez la distribution :",
    ("Avant le préprocessing de la tranche d'âge", "Après le préprocessing de la tranche d'âge")
      )
  if option == "Avant le préprocessing de la tranche d'âge":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='age', data=df_avprepro_age, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Tranche d'âge")
    st.pyplot(plt)
  elif option == "Après le préprocessing de la tranche d'âge":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='age_group', data=df_apprepro_age, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Tranche d'âge")
    plt.xticks([0, 1, 2, 3,4,5,6, 7], ['0 à 10', '11 à 20', "21-30", "31 à 40", "41 à 50", "51 à 60", "61 à 70", "71 et plus"])
    st.pyplot(plt)
  st.write("-------------")
  
  st.write("####  Regroupement de variables")
  st.write("##### Modes de déplacement") 
  df_avprepro_mdp = df_ana.copy()
  df_apprepro_mdp = df_model.copy()
  option = st.selectbox(
    "Choisissez la distribution :",
    ("Avant le préprocessing du mode de déplacement", "Après le préprocessing du mode de déplacement")
      )
  if option == "Avant le préprocessing du mode de déplacement":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='catv', data=df_avprepro_mdp, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Mode de déplacement")
    st.pyplot(plt)
  elif option == "Après le préprocessing du mode de déplacement":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='catv', data=df_apprepro_mdp, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Mode de déplacement")
    plt.xticks([0,1, 2, 3,4,5,6], ['Marche', 'EDPM', "Vélo", "2RM", "VT", "VU", "Voiture"])
    st.pyplot(plt)
  st.write("-------------")
  
  st.write("##### Systèmes de sécurité") 
  fig = plt.figure(figsize=(5,5))
  df_avprepro_se = df_ana.copy()
  df_apprepro_se = df_model.loc[(df_model['secu1'] >= 1)].copy()
  option = st.selectbox(
    "Choisissez la distribution :",
    ("Avant le préprocessing du système de sécurité", "Après le préprocessing du système de sécurité")
      )
  if option == "Avant le préprocessing du système de sécurité":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='secu1', data=df_avprepro_se, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Système de sécurité")
    st.pyplot(plt)
  elif option == "Après le préprocessing du système de sécurité":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='secu1', data=df_apprepro_se, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Système de sécurité")
    plt.xticks([0,1], ['Présence', 'Absence'])
    st.pyplot(plt)
  st.write("-------------")
    
  st.write("##### Variable cible : gravité") 
  df_avprepro_se = df_ana.copy()
  df_apprepro_se = df_model.copy()
  option = st.selectbox(
    "Choisissez la distribution :",
    ("Avant le préprocessing de gravité", "Après le préprocessing de gravité")
      )
  if option == "Avant le préprocessing de gravité":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='grav', data=df_avprepro_se, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xlabel("Gravité")
    plt.xticks([0,1, 2, 3,4], ['Non renseigné', 'Indemne', "Tué", "Blessé hospitalisé", "Blessé léger"])
    st.pyplot(plt)
  elif option == "Après le préprocessing de gravité":
    plt.figure(figsize=(8, 4))
    sns.countplot(x='grav', data=df_apprepro_se, palette='pastel')
    plt.ylabel("Nombre d'accidents")
    plt.xticks([0,1], ['Accident sans gravité', "Accidents avec gravité"])

    st.pyplot(plt)
    
  st.write("Variables d'origines :", "1 – Indemne, 2 – Tué, 3 – Blessé hospitalisé, 4 – Blessé léger")
  st.write("Nouvelles variables  :", "1 – Accident sans gravité, 2 – Accident avec gravité")    
  

  st.write("#### Suppression des variables") 
  df_mqt = pd.DataFrame({
    'Variables': ['dep', 'mois', "heure, minute", "jour, date_obj","an_nais, age","an", "lartpc, occutc, v2", "id_accident, id_vehicule, id_usager"],
    'Raison': ['doublon avec les variables lat et long', 'doublon avec weekday','doublon avec tranche horaire', "doublon avec weekday",'doublon avec tranche âge', "année de l'accident", "90 % de manquants", "variables qui a servi à faire la liaison"]
})
  st.table(df_mqt)
  
  if st.checkbox("Afficher le dataframe final") :
    st.dataframe(df_model.head())
  
############################################### Modélisation ###################################################   
 
if page == pages[4] : 
  st.write("#### Sélection du Modèle de Machine Learning")   
  
  df_model['index'] = df_model.index
  df_train, df_test = train_test_split(df_model, test_size=0.2, random_state=42)
  
  X_train = df_train.drop(["grav","index"], axis=1)
  y_train = df_train.grav
  X_test = df_test.drop(["grav","index"], axis=1)
  y_test = df_test.grav
     
  scaler = StandardScaler()

  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test) 
  
  models = {
    "model adaboost" : joblib.load('models_saved/best_estimator_ada_model.joblib'),
  #  "model extra trees classifier" : joblib.load('Brouillons/extra_tree_classifier_model.joblib'),
    "model xgbc" : joblib.load('models_saved/GBClassifier_model.joblib'),
  #  "model régression logistique" : joblib.load('models_saved/lr_model.joblib'),
    "model lgbm" : joblib.load('models_saved/lgbm_model.joblib'),
    "model random forest" : joblib.load('Brouillons/rf_model.joblib')#,
  #  "model knn" : joblib.load('models_saved/KNeighborsClassifier_model.joblib')
           }
  selected_model = st.selectbox("Choisissez un modèle:", options=list(models.keys()))
  model = models[selected_model]
  y_pred = model.predict(X_test_scaled)
  st.write(f"### Résultats pour {selected_model}")
  
  st.write("#### Rapport de Classification")
  st.text(classification_report(y_test, y_pred))
  st.write("-------------")
  
  
  st.write("#### Matrice de Confusion en pourcentages")
  cm = confusion_matrix(y_test, y_pred)
  cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
  
  fig, ax = plt.subplots(figsize=(8, 6))
  disp.plot(ax=ax, cmap='Blues', values_format=".2f")
  st.pyplot(fig)
  st.write("-------------")
  
  st.write("#### Importances des variables")
  features = X_test.columns
  importances = model.feature_importances_
  indices = np.argsort(importances)

  fig = plt.figure(figsize=(10, 6))
  plt.barh(range(len(indices)), importances[indices], align="center")
  plt.yticks(range(len(indices)), [features[i] for i in indices])
  plt.xlabel('Importance Relative')
  st.pyplot(fig)
  st.write("-------------")
  
  st.write("#### Carte des erreurs du model")
  df_test['y_pred'] = y_pred
  df_test = df_test[['index', 'grav', 'y_pred','lat','long']]
  df_test = df_test.rename(columns={'grav': 'y_test', 'lat': 'latitude', 'long': 'longitude'})
  df_mismatch = df_test[df_test['y_test'] != df_test['y_pred']]
  st.write(f"Nombre d'erreurs : {len(df_mismatch)}")
  st.write(f"Nombre d'accidents : {len(y_test)}")
  prt_er = (1-accuracy_score(y_test, y_pred)) * 100
  st.write(f"Pourcentage : {prt_er: .2f}%")
  st.map(df_mismatch, size=10)
  
  ############################################### Perspectives ###################################################   
 
if page == pages[5] : 
  st.write("#### Bilan")  
  st.write('La mission du projet était de prédire si un accident est grave ou non en fonction des données historiques.')
  st.write("Nous sommes passés par plusieurs étapes, allant de l’exploration du dataset, à sa visualisation jusqu'à l'implémentation d'une solution de prédictions dans application web. ")
  
  st.write("#### Perspectives")
  st.write("Avec ce projet, nous avons pu effectuer tout d’abord une modélisation simple avec des modèles de classification comme le Random Forest Classifier. Puis nous avons effectuer d'autres modélisations en Machine Learning comme le Adaboost Classifier, le LightGBM Classifier ou le XGBoost Classifier.")
  st.write("Pour aller plus loin, nous pourrions effectuer une énième modélisation avec les réseaux récurrents RNN pour comparer les résultats avec les autres modèles de Machine Learning.")
  st.write("Il serait possible également de développer un modèle de scoring des zones à risques en fonction des informations météorologiques, l’emplacement géographique (coordonnées GPS, images satellite, …). Cela permettrait de cibler les campagnes de prévention en fonction des zones à risques.")
  
  
  