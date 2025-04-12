Documentation du projet CBIR avec Authentification Multiple
Description
Cette application web implémente un système de recherche d'images basée sur le contenu (CBIR) avec authentification multi-méthodes. Elle permet aux utilisateurs de s'inscrire et de se connecter via plusieurs méthodes (mot de passe, reconnaissance faciale, Google, Facebook) et d'effectuer des recherches d'images similaires en utilisant différents descripteurs et mesures de distance.
Fonctionnalités principales
Authentification

Connexion par nom d'utilisateur et mot de passe
Connexion par reconnaissance faciale
Authentification OAuth (Google, Facebook)
Gestion de profil utilisateur

Recherche d'images (CBIR)

Descripteurs d'images:

GLCM (Gray-Level Co-occurrence Matrix)
Haralick
BiT (Bio-Inspired Texture) simplifié
Concaténation des descripteurs


Mesures de distance:

Euclidienne
Manhattan
Tchebychev (Chebyshev)
Canberra


Extraction et sauvegarde de signatures d'images
Interface utilisateur pour la recherche et l'affichage des résultats

Structure du projet
projet/
├── app.py                  # Application principale
├── cbir_functions.py       # Fonctions pour la recherche d'images
├── animalsCbir/            # Dataset d'images organisé par catégories
│   ├── cat/
│   ├── dog/
│   └── ...
├── signatures/             # Fichiers de signatures générés
│   ├── SignaturesGlcm.npy
│   ├── SignaturesHaralick.npy
│   ├── SignaturesBit.npy
│   └── SignaturesConcat.npy
└── temp/                   # Dossier temporaire pour les images téléversées
Modules et dépendances

streamlit: Interface utilisateur web
face_recognition: Reconnaissance faciale
pymongo: Base de données MongoDB
opencv-python (cv2): Traitement d'images
numpy: Manipulation de tableaux numériques
skimage: Extraction de caractéristiques GLCM
mahotas: Extraction de caractéristiques Haralick
scipy: Calcul de distances

Architecture technique
Module d'authentification

Stockage sécurisé des identifiants (hachage des mots de passe)
Intégration OAuth pour Google et Facebook
Capture et encodage des visages pour l'authentification biométrique

Module CBIR

Extraction et stockage des signatures d'images
Implémentation de différents descripteurs visuels
Algorithmes de comparaison et de mesure de similarité
Interface de recherche et de visualisation des résultats

Utilisation
Installation
bash# Cloner le dépôt
git clone https://github.com/Moutail/cbir-authentification.git
cd cbir-authentification

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
Configuration

Créer un compte MongoDB et configurer la variable MONGODB_URI
Organiser les images dans des sous-dossiers de animalsCbir/ par catégorie
Configurer les identifiants OAuth pour Google et Facebook si nécessaire

Premier lancement

Créer un compte utilisateur
Se connecter à l'application
Accéder à la page "Recherche d'Images"
Extraire les signatures (bouton "Extraire toutes les signatures")
Téléverser une image et configurer les paramètres de recherche
Visualiser les résultats

Algorithmes clés
Extraction de caractéristiques GLCM
pythondef glcm(image_path):
    img = cv2.imread(image_path, 0)
    co_matrice = graycomatrix(img, [1], [np.pi/2], None, symmetric=False, normed=False)
    contrast = graycoprops(co_matrice, 'contrast')[0, 0]
    # ... autres propriétés
    features = [contrast, dissimilarity, homogeneity, correlation, energy, ASM]
    return features
Recherche d'images similaires
pythondef recherche_images(bdd_signature, caracteristique_requete, distance_type, K):
    img_similaire = []
    for instance in bdd_signature:
        carac, label, img_chemin = instance[:-2], instance[-2], instance[-1]
        # Calcul de distance selon la méthode choisie
        img_similaire.append((img_chemin, dist, label))
    img_similaire.sort(key=lambda x: x[1])
    return img_similaire[:K]  # Retourne les K plus similaires
Crédits
Ce projet se base sur les techniques de vision par ordinateur et de reconnaissance d'images étudiées dans le cadre du cours IA2: Vision artificielle et reconnaissance de formes (420-1AB-TT).

Exemple de README pour GitHub
markdown# CBIR avec Authentification Multiple

Application web de recherche d'images basée sur le contenu (CBIR) avec système d'authentification multimodal.

## Fonctionnalités

- **Authentification multi-méthodes**:
  - Email/Mot de passe
  - Reconnaissance faciale
  - OAuth (Google, Facebook)

- **Recherche d'images**:
  - Descripteurs GLCM, Haralick, et BiT
  - Plusieurs mesures de distance (Euclidienne, Manhattan, Tchebychev, Canberra)
  - Interface utilisateur intuitive pour la recherche et la visualisation

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/cbir-authentification.git
cd cbir-authentification

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
Configuration

Créez une base de données MongoDB et configurez l'URI dans le code
Organisez vos images dans des sous-dossiers par catégorie dans animalsCbir/
Configurez vos identifiants OAuth pour Google et Facebook

Usage

Inscrivez-vous et connectez-vous à l'application
Extrayez les signatures des images
Téléversez une image requête
Ajustez les paramètres de recherche
Explorez les résultats!

Technologies

Streamlit
OpenCV
Scikit-image
MongoDB
Face Recognition

Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
