# CBIR avec Authentification Multiple

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
