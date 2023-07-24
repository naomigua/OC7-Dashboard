# DASHBOARD pour le projet de détection/prédiction de faillite bancaire

## Objectif du Dashboard / Application


## Description
Ce Dashboard qui tourne sur Streamlit permet d'afficher les données clients qui permettent de prédire le risque de faillite bancaire.
Elle est à destination des conseillers clientèle qui pourront visualiser le score dudit risque, les variables qui ont le plus contribué au score. Cet application a 3 volets :
- Tableau clientèle #permet d'afficher l'ensemble des clients et leurs attributs
- Comparaison clientèle #permet de comparer un client donné aux clients qui lui ressemblent le plus (via un knn - sélection du nombre de clients pour la comparaison en entrée); la compairaison se fait sur le type de revenu, d'éducation, de logement, sur le genre, le nombre d'enfants ainsi qu'une source externe de notation
- Visualisation score #retourne le score correspondant au client sélectionné ainsi qu'une visualisation qui montre quelles sont les variables ayant le plus contribué au score. Cette page requête une API en lui envoyant les données clients et restitue les résultats.
## Découpage des dossiers
- DASHBOARD/  
  - bin
    - df_dash.csv #données
    - logopretadepenser.JPG 
  - .gitignore
  - dashboard.py # code du dashboard
  - Procfile # pour déploiement Heroku
  - README.md # fichier descriptif
  - requirements.txt # liste des packages utilisés
  - runtime.txt # version de python utilisée
  - setup.sh # fichier de setup streamlit

## Librairies utilisées
Cet API fonctionne sur python-3.11.4 et nécessite :
- fastapi==0.100.0
- matplotlib==3.7.2
- numpy==1.24.4
- pandas==2.0.3
- Requests==2.31.0
- scikit_learn==1.3.0
- seaborn==0.12.2
- shap==0.42.0
- streamlit==1.24.1
- uvicorn==0.22.0
- gunicorn==20.1.0

## Utilisation
Pour la lancer en local il faut lancer la commande : 
streamlit run dashboard.py 
et remplace l'adresse d'API par l'adresse locale dans la fonction visualisation score
