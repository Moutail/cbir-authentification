import streamlit as st
import numpy as np
import cv2
import os
import time
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from scipy.spatial import distance

DATASET_PATH = "./animalsCbir/"
SIGNATURES_PATH = "./signatures/"


def glcm(image_path):
    """Extraction des caractéristiques GLCM"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
            
        co_matrice = graycomatrix(img, [1], [np.pi/2], None, symmetric=False, normed=False)
        contrast = graycoprops(co_matrice, 'contrast')[0, 0]
        dissimilarity = graycoprops(co_matrice, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(co_matrice, 'homogeneity')[0, 0]
        correlation = graycoprops(co_matrice, 'correlation')[0, 0]
        energy = graycoprops(co_matrice, 'energy')[0, 0]
        ASM = graycoprops(co_matrice, 'ASM')[0, 0]
        features = [contrast, dissimilarity, homogeneity, correlation, energy, ASM]
        features = [float(x) for x in features]
        return features
    except Exception as e:
        st.error(f"Erreur lors de l'extraction GLCM: {str(e)}")
        return [0.0] * 6 

def haralik_feat(image_path):
    """Extraction des caractéristiques Haralick"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
            
        features = haralick(img).mean(0).tolist()
        features = [float(x) for x in features]
        return features
    except Exception as e:
        st.error(f"Erreur lors de l'extraction Haralick: {str(e)}")
        return [0.0] * 13 

def simple_bit(image_path):
    """Remplacement simplifié du descripteur BiT"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
        
        mean = np.mean(img)
        std = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        hist = np.histogram(img, bins=10, range=(0, 256))[0]
        hist = hist / np.sum(hist) 
        
     
        features = [mean, std, min_val, max_val]
        features.extend(hist)
        

        if len(features) < 16:
            skewness = np.mean(((img - mean) / std) ** 3) if std > 0 else 0
            kurtosis = np.mean(((img - mean) / std) ** 4) if std > 0 else 0
            features.extend([skewness, kurtosis])
            features.extend([0.0] * (16 - len(features)))  
        
        return features[:16]  
    except Exception as e:
        st.error(f"Erreur lors de l'extraction simple BiT: {str(e)}")
        return [0.0] * 16

def concat(image_path):
    """Concaténation des trois descripteurs"""
    try:
        return glcm(image_path) + haralik_feat(image_path) + simple_bit(image_path)
    except Exception as e:
        st.error(f"Erreur lors de la concaténation des descripteurs: {str(e)}")
        return [0.0] * 35 

# Fonctions de calcul de distance
def manhattan_distance(v1, v2):
    """Distance de Manhattan"""
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1 - v2))
    return dist

def euclidean_distance(v1, v2):
    """Distance Euclidienne"""
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist

def chebyshev_distance(v1, v2):
    """Distance de Tchebychev"""
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1 - v2))
    return dist

def canberra_distance(v1, v2):
    """Distance de Canberra"""
    return distance.canberra(v1, v2)

# Fonction de recherche d'images similaires
def recherche_images(bdd_signature, caracteristique_requete, distance_type, K):
    """Recherche les K images les plus similaires"""
    img_similaire = []
    
    for instance in bdd_signature:
        carac, label, img_chemin = instance[:-2], instance[-2], instance[-1]
        
        if distance_type == 'manhattan':
            dist = manhattan_distance(caracteristique_requete, carac)
        elif distance_type == 'euclidean':
            dist = euclidean_distance(caracteristique_requete, carac)
        elif distance_type == 'chebyshev':
            dist = chebyshev_distance(caracteristique_requete, carac)
        elif distance_type == 'canberra':
            dist = canberra_distance(caracteristique_requete, carac)
        else:
            dist = euclidean_distance(caracteristique_requete, carac)  # Par défaut
        
        img_similaire.append((img_chemin, dist, label))
    
    img_similaire.sort(key=lambda x: x[1])
    return img_similaire[:K]

def extraction_signatures(chemin_dossier, type_descripteur):
    """Extrait les signatures pour toutes les images du dataset"""
    st.info(f"Extraction des signatures {type_descripteur} en cours...")
    progress_bar = st.progress(0)
    
    list_carac = []
    total_files = 0
    processed_files = 0
    
    # Compter le nombre total de fichiers
    for root, dirs, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                total_files += 1
    
    for root, dirs, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                relative_path = os.path.relpath(os.path.join(root, file), chemin_dossier)
                path = os.path.join(root, file)
                
                try:
                    if type_descripteur == 'glcm':
                        caracteristiques = glcm(path)
                    elif type_descripteur == 'haralick':
                        caracteristiques = haralik_feat(path)
                    elif type_descripteur == 'bit':
                        caracteristiques = simple_bit(path)
                    else:  # 'concat' par défaut
                        caracteristiques = concat(path)
                    
                    class_name = os.path.dirname(relative_path)
                    list_carac.append(caracteristiques + [class_name, relative_path])
                except Exception as e:
                    st.warning(f"Erreur lors du traitement de {path}: {str(e)}")
                
                processed_files += 1
                progress_bar.progress(processed_files / total_files if total_files > 0 else 0)
    
    Signatures = np.array(list_carac)
    signature_file = os.path.join(SIGNATURES_PATH, f"Signatures{type_descripteur.capitalize()}.npy")
    np.save(signature_file, Signatures)
    
    st.success(f"Extraction terminée. {processed_files} images traitées.")
    return signature_file

# Fonction pour télécharger une image temporaire
def save_uploaded_image(uploaded_file):
    """Sauvegarde une image téléversée dans un répertoire temporaire"""
    if uploaded_file is not None:
        # Créer un répertoire temporaire si nécessaire
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Sauvegarder le fichier
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    return None

def cbir_page():
    """Page de recherche d'images basée sur le contenu"""
    st.header("Recherche d'Images basée sur le Contenu (CBIR)")
    
    # Vérifier si les signatures ont été extraites
    signatures_files = {
        'GLCM': os.path.join(SIGNATURES_PATH, "SignaturesGlcm.npy"),
        'Haralick': os.path.join(SIGNATURES_PATH, "SignaturesHaralick.npy"),
        'BiT': os.path.join(SIGNATURES_PATH, "SignaturesBit.npy"),
        'Concaténation': os.path.join(SIGNATURES_PATH, "SignaturesConcat.npy")
    }
    
    missing_signatures = [name for name, path in signatures_files.items() 
                          if not os.path.exists(path)]
    
    # Afficher un avertissement si des signatures sont manquantes
    if missing_signatures:
        st.warning(f"Les signatures suivantes n'ont pas été extraites: {', '.join(missing_signatures)}")
        
        with st.expander("Extraction des signatures"):
            st.write("Vous devez extraire les signatures avant de pouvoir utiliser la recherche CBIR.")
            
            if st.button("Extraire toutes les signatures"):
                for desc_type in ['glcm', 'haralick', 'bit', 'concat']:
                    extraction_signatures(DATASET_PATH, desc_type)
                st.success("Toutes les signatures ont été extraites.")
                st.rerun()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres de recherche")
        
        # Upload d'image
        uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Afficher l'image téléversée
            image_path = save_uploaded_image(uploaded_file)
            st.image(uploaded_file, caption="Image requête", width=250)
            
            # Options de recherche
            descripteur = st.selectbox(
                "Descripteur à utiliser",
                ["GLCM", "Haralick", "BiT", "Concaténation"],
                index=3  # Concaténation par défaut
            )
            
            distance_type = st.selectbox(
                "Mesure de distance",
                ["euclidean", "manhattan", "chebyshev", "canberra"],
                index=0
            )
            
            k_images = st.slider("Nombre d'images similaires (K)", min_value=1, max_value=20, value=10)
            
            # Bouton de recherche
            if st.button("Rechercher des images similaires"):
                # Déterminer le fichier de signatures à utiliser
                desc_map = {
                    "GLCM": "Glcm",
                    "Haralick": "Haralick",
                    "BiT": "Bit",
                    "Concaténation": "Concat"
                }
                
                signatures_file = os.path.join(SIGNATURES_PATH, f"Signatures{desc_map[descripteur]}.npy")
                
                if not os.path.exists(signatures_file):
                    st.error(f"Le fichier de signatures {signatures_file} n'existe pas.")
                else:
                    # Charger les signatures
                    signatures = np.load(signatures_file)
                    
                    # Extraire les caractéristiques de l'image requête
                    if descripteur == "GLCM":
                        requete_features = glcm(image_path)
                    elif descripteur == "Haralick":
                        requete_features = haralik_feat(image_path)
                    elif descripteur == "BiT":
                        requete_features = simple_bit(image_path)
                    else:  # Concaténation par défaut
                        requete_features = concat(image_path)
                    
                    # Rechercher les images similaires
                    results = recherche_images(signatures, requete_features, distance_type, k_images)
                    
                    # Afficher les résultats
                    st.success(f"{len(results)} images similaires trouvées!")
                    
                    # Afficher les résultats dans la colonne de droite
                    with col2:
                        st.subheader("Résultats de la recherche")
                        
                        # Créer une grille pour afficher les résultats
                        n_cols = 3
                        n_rows = (len(results) + n_cols - 1) // n_cols
                        
                        for row in range(n_rows):
                            cols = st.columns(n_cols)
                            for col in range(n_cols):
                                idx = row * n_cols + col
                                if idx < len(results):
                                    img_path, distance, label = results[idx]
                                    full_path = os.path.join(DATASET_PATH, img_path)
                                    
                                    try:
                                        img = cv2.imread(full_path)
                                        if img is not None:
                                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                            cols[col].image(img, caption=f"{label} (d={distance:.4f})", width=150)
                                        else:
                                            cols[col].error(f"Impossible de charger l'image: {img_path}")
                                    except Exception as e:
                                        cols[col].error(f"Erreur: {str(e)}")
        else:
            with col2:
                st.info("Téléversez une image et configurez les paramètres de recherche pour trouver des images similaires.")