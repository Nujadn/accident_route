Projet : Accident de la route
==============


## Présentation du projet

Le projet « Accidents routiers en France » s’inscrit dans le cadre du cursus Datascientist, formation
continue proposée par l’école Datascientest.
Il offre l’avantage de faire appel à l’ensemble des notions fondamentales pour le traitement d’un projet
de Datascience. Ce sujet d’actualité étant récurrent, il est intéressant de pouvoir mettre en exergue les
points sensibles les plus accidentogènes. Cela peut constituer une bonne démarche pour parvenir à
limiter le nombre de victimes et de matériel endommagé.

Afin de pouvoir prédire la gravité des accidents de la route en France, nous nous sommes
principalement basés sur les datasets fournis par [le Ministère de l’intérieur et des Outre-Mer](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/#/resources). Les
bases de données sont réparties annuellement en 4 fichiers au format csv, de 2005 à 2022 :
- Caractéristiques
- Lieux
- Véhicules
- Usagers

L'objectif principal est de développer une application capable d'analyser et de prédire les accidents de la route.

Ce projet met en œuvre des compétences en Data Science et Machine Learning, en intégrant des modèles avancés pour l'exploration et la visualisation des données.

## Objectifs du Projet

- Développer une application Streamlit interactive pour la visualisation et la prédiction des résultats.
- Appliquer des techniques de traitement des données et de Machine Learning.
- Déployer une interface utilisateur simple pour les analyses et les prédictions en temps réel.

## Membres de l'Équipe

- [Sarah Vandendriessche](https://github.com/Nujadn) (lien github)
- [Mohand Amechtouh Belkaïd](https://github.com/MoBEL3) (lien github)
- [Thierry Famboupe Taffou](https://github.com/Famboupe) (lien github)
- [Elisabeth Miketa Kalala](https://github.com/ElKal7) (lien github)

## Installation

Pour installer et exécuter ce projet sur votre machine locale, suivez les étapes suivantes :

1. Cloner ce dépôt :
   ```bash
   git clone https://github.com/votrenomdutilisateur/votre-nom-repo.git
   cd votre-nom-repo
    ```

2. Créer un environnement virtuel :
   ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows utilisez `venv\Scripts\activate`
    ```

3. Installer les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ``` 

## Étapes pour exécuter l'application

Après avoir exécuter les étapes précentes, vous pouvez lancer l'application en utilisant la commande suivante :

```bash
streamlit run streamlit_final_V4.py
```

Pour accéder l'application : une fois l'application démarrée, elle sera accessible via votre navigateur à l'adresse suivante : localhost:8501.