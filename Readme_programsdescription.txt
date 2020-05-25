les datasets brutes sont accessibles sur: https://www.kaggle.com/c/home-credit-default-risk/data
il y a 3 fichiers jupyter.
Description des colones.ipynb explore la decription des features des datasets
Le nettoyage, l'assemblage et le feature engineering des donnees brutes est effectuees dans Nb1.ipynb 
La conception du modele machine learning est effectue dans le notebook Nb2.ipynb
Procfile est un fichier qui specifie a heroku quelle application lancer, ici app11.py qui contient le dashboard
requirements.txt est un fichier texte que lit heroku pour savoir quelles librairies doivent etre charge 
pour le bon fonctionnement de app11.py
Le dataset des donnees test pour effectuer la prediction est df_test_red_dash.csv
Ce dataset est charge dans app11.py
Le dataset des seuils et des poids des features sont df_xgb_limit.csv et df_coef_xgboo.csv
bootstrap.min.css contient les casading style sheets de bootstrap

