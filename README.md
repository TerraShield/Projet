Introduction
Ce logiciel permet de regrouper et analyser les lettrines dans les anciens documents grâce à des algorithmes de clustering.

Fonctionnalités principales :
-Chargement d’images.
-Analyse des lettrines via des algorithmes de clustering.
-Visualisation des résultats, y compris des heatmaps et histogrammes.

Processus installation de l'environnement virtuel:

Installer Miniconda (https://docs.anaconda.com/miniconda/install/)
- Windows :
    - Lancez le .exe
    - Suivez les instructions 
    ( Ne pas oublier de rajouter Miniconda à la variable PATH )
- MacOs/Linux :
    - Dans un terminal, exécutez la commande suivante à l'endroit où est le fichier d'installation :
        " bash Miniconda3-latest-Linux-x86_64.sh " / " bash Miniconda3-latest-MacOSX-arm64.sh "
    - Suivez les instructions

Vous pouvez vérifier par la suite que conda est bien installé avec la commande suivante :
    " conda --version "

Après l'installation, il est recommandé de mettre Conda à jour
Pour se faire exécutez la commande suivante :
    " conda update -n base -c default conda "

Par la suite, avec un environnement déjà installé, ici "projet" issue du fichier environnement.yml, vous pouvez l'initialiser en exécutant ces deux commandes :
    " conda env create -f environnement.yml " 
    " conda activate projet "

Instructions d’utilisation
Lancer l’application :

-Lancer le fichier main.py
-Choisir un dossier d’images :
Cliquez sur "Sélectionner un dossier" dans le menu principal.
Naviguez jusqu’au dossier contenant les images des lettrines.
-Choisir un onglet :
	-Liste :
		Visualisez toutes les images chargées, organisées par sous-dossier.
	-Clustering :
		Sélectionnez un algorithme (KMeans ou DBSCAN).
Indiquez le nombre de clusters (ex :5)
		Cliquez sur Appliquer pour afficher les regroupements.
	-Histogramme :
		Visualisez l’histogramme des couleurs pour chaque image
	-Heatmap :
		Obtenez une vue heatmap basée sur les caractéristiques SIFT.

Structure des fichiers
main.py : Point d’entrée pour exécuter l’application.
menu.py : Gestion du menu principal et navigation entre les onglets.
cluster_page.py : Implémentation des algorithmes de clustering.
heatmap_algo.py : Traitement des images avec SIFT et BoW.
heatmap_page.py : Affichage des heatmaps.
histogramme.py : Calcul et affichage des histogrammes de couleurs.
detail_image.py : Affichage détaillé d’une image sélectionnée.

Algorithmes utilisés
-Clustering (KMeans et DBSCAN) :
KMeans regroupe les images en fonction de leur proximité dans un espace de caractéristiques.
DBSCAN identifie les clusters denses et marque les anomalies. 
ex: {"eps": 0.2, "min_samples": 3}
eps : La distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.
min_samples : Le nombre minimal de points dans un voisinage pour qu'un point soit considéré comme un noyau.

-Réduction de dimensionnalité (PCA) :
Utilisée pour réduire les dimensions avant clustering.
-SIFT et BoW :
SIFT (Scale-Invariant Feature Transform) extrait des points clés des images.
BoW (Bag of Words) regroupe ces descripteurs en mots visuels.

Fonctionnement de l'interface
-Créée avec Tkinter.
-Navigation :
Les onglets permettent d’accéder aux différentes fonctionnalités.
-Paramètres :
Menu déroulant et champs de saisie pour configurer les algorithmes.

Librairies utilisées :
-numpy
-scikit-learn
-scikit-image
-matplotlib
-Pillow
-tkinter
-opencv-python-headless
-mplcursors
