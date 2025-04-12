import streamlit as st
import face_recognition
import numpy as np
import cv2
import os
import time
import hashlib
import uuid
import requests
import json
from urllib.parse import urlencode
import pymongo
from bson.binary import Binary
import pickle

from cbir_functions import (
    glcm, haralik_feat, simple_bit, concat, 
    manhattan_distance, euclidean_distance, chebyshev_distance, canberra_distance,
    recherche_images, extraction_signatures, save_uploaded_image, cbir_page
)
# Configuration de la page
st.set_page_config(page_title="Authentification avec Reconnaissance Faciale", layout="wide")

# Chemin pour le dataset et les signatures
DATASET_PATH = "./animalsCbir/"
SIGNATURES_PATH = "./signatures/"

# Configuration OAuth
GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""
FACEBOOK_APP_ID = ""
FACEBOOK_APP_SECRET = ""
REDIRECT_URI = ""

# Configuration MongoDB
MONGODB_URI = ""

# Connexion à MongoDB
@st.cache_resource
def get_database():
    client = pymongo.MongoClient(MONGODB_URI)
    return client["cbir-auth-app"]

# Initialisation des variables de session
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'auth_method' not in st.session_state:
    st.session_state.auth_method = None
if 'oauth_state' not in st.session_state:
    st.session_state.oauth_state = str(uuid.uuid4())
if 'oauth_provider' not in st.session_state:
    st.session_state.oauth_provider = None
if 'redirect_page' not in st.session_state:
    st.session_state.redirect_page = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Fonctions utilitaires pour MongoDB
def get_user_collection():
    db = get_database()
    return db["users"]

def get_user_by_username(username):
    users = get_user_collection()
    return users.find_one({"username": username})

def get_user_by_google_id(google_id):
    users = get_user_collection()
    return users.find_one({"auth_methods.google_id": google_id})

def get_user_by_facebook_id(facebook_id):
    users = get_user_collection()
    return users.find_one({"auth_methods.facebook_id": facebook_id})

def get_user_by_email(email):
    users = get_user_collection()
    return users.find_one({"email": email})

def insert_user(user_data):
    users = get_user_collection()
    return users.insert_one(user_data)

def update_user(username, update_data):
    users = get_user_collection()
    return users.update_one({"username": username}, {"$set": update_data})

def get_all_users_with_face():
    users = get_user_collection()
    return list(users.find({"has_face_encoding": True}))

def hash_password(password):
    """Hacher un mot de passe"""
    return hashlib.sha256(password.encode()).hexdigest()

def capture_face():
    """Capturer une image depuis la webcam et extraire l'encodage facial"""
    st.info("Placez votre visage devant la caméra et regardez droit.")
    
    img_file = st.camera_input("Prenez une photo de votre visage")
    
    if img_file is not None:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_img)
        
        if len(face_locations) == 0:
            st.error("Aucun visage détecté! Veuillez réessayer.")
            return None
        elif len(face_locations) > 1:
            st.error("Plusieurs visages détectés! Un seul visage est nécessaire.")
            return None
        
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if face_encodings:
            # Dessiner un rectangle autour du visage
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Afficher l'image avec le rectangle
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Visage détecté!")
            
            return face_encodings[0]
    
    return None

def get_google_auth_url():
    """Générer l'URL d'authentification Google"""
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'state': st.session_state.oauth_state,
        'access_type': 'offline',
        'prompt': 'consent'
    }
    return f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"

def get_facebook_auth_url():
    """Générer l'URL d'authentification Facebook"""
    params = {
        'client_id': FACEBOOK_APP_ID,
        'redirect_uri': REDIRECT_URI,
        'state': st.session_state.oauth_state,
        'scope': 'email,public_profile'
    }
    return f"https://www.facebook.com/v13.0/dialog/oauth?{urlencode(params)}"

def process_oauth_callback():
    """Traiter le callback OAuth"""
    query_params = st.query_params
    
    if 'code' in query_params and 'state' in query_params:
        code = query_params['code']
        state = query_params['state']
        
        # Vérifier que l'état correspond pour éviter les attaques CSRF
        if state != st.session_state.oauth_state:
            st.error("État OAuth invalide. Tentative d'attaque potentielle.")
            return False, None
        
        # Déterminer le fournisseur en fonction des paramètres
        if 'error' in query_params:
            st.error(f"Erreur d'authentification: {query_params['error']}")
            return False, None
        
        # Obtenir un token en fonction du code
        provider = st.session_state.oauth_provider
        
        if provider == 'google':
            return exchange_google_code(code)
        elif provider == 'facebook':
            return exchange_facebook_code(code)
    
    return False, None

def exchange_google_code(code):
    """Échanger le code d'autorisation Google contre un token d'accès"""
    try:
        token_url = 'https://oauth2.googleapis.com/token'
        data = {
            'code': code,
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'redirect_uri': REDIRECT_URI,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(token_url, data=data)
        token_data = response.json()
        
        if 'access_token' not in token_data:
            st.error(f"Erreur lors de l'échange du code: {token_data.get('error_description', 'Inconnu')}")
            return False, None
        
        # Obtenir les informations de l'utilisateur
        user_info_url = 'https://www.googleapis.com/oauth2/v1/userinfo'
        headers = {'Authorization': f"Bearer {token_data['access_token']}"}
        user_response = requests.get(user_info_url, headers=headers)
        user_info = user_response.json()
        
        if 'id' not in user_info:
            st.error("Impossible d'obtenir les informations de l'utilisateur")
            return False, None
        
        # Chercher l'utilisateur par Google ID
        existing_user = get_user_by_google_id(user_info['id'])
        
        # Si non trouvé, chercher par email
        if not existing_user and 'email' in user_info:
            existing_user = get_user_by_email(user_info['email'])
            
            if existing_user:
                # Ajouter l'authentification Google
                update_user(existing_user['username'], {
                    "auth_methods.google": True,
                    "auth_methods.google_id": user_info['id']
                })
        
        # Si toujours pas trouvé, créer un nouvel utilisateur
        if not existing_user:
            username = f"google_{user_info.get('name', '').replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}"
            new_user = {
                'username': username,
                'email': user_info.get('email', ''),
                'auth_methods': {
                    'google': True,
                    'google_id': user_info['id']
                },
                'has_face_encoding': False
            }
            insert_user(new_user)
            existing_user = get_user_by_username(username)
        
        # Authentifier l'utilisateur
        st.session_state.authenticated = True
        st.session_state.current_user = existing_user['username']
        st.session_state.auth_method = 'google'
        
        return True, existing_user['username']
        
    except Exception as e:
        st.error(f"Erreur lors de l'authentification Google: {str(e)}")
        return False, None

def exchange_facebook_code(code):
    """Échanger le code d'autorisation Facebook contre un token d'accès"""
    try:
        token_url = 'https://graph.facebook.com/v13.0/oauth/access_token'
        params = {
            'client_id': FACEBOOK_APP_ID,
            'client_secret': FACEBOOK_APP_SECRET,
            'redirect_uri': REDIRECT_URI,
            'code': code
        }
        
        response = requests.get(token_url, params=params)
        token_data = response.json()
        
        if 'access_token' not in token_data:
            st.error(f"Erreur lors de l'échange du code: {token_data.get('error', {}).get('message', 'Inconnu')}")
            return False, None
        
        # Obtenir les informations de l'utilisateur
        user_info_url = 'https://graph.facebook.com/me'
        params = {
            'fields': 'id,name,email',
            'access_token': token_data['access_token']
        }
        user_response = requests.get(user_info_url, params=params)
        user_info = user_response.json()
        
        if 'id' not in user_info:
            st.error("Impossible d'obtenir les informations de l'utilisateur")
            return False, None
        
        # Chercher l'utilisateur par Facebook ID
        existing_user = get_user_by_facebook_id(user_info['id'])
        
        # Si non trouvé, chercher par email
        if not existing_user and 'email' in user_info:
            existing_user = get_user_by_email(user_info['email'])
            
            if existing_user:
                # Ajouter l'authentification Facebook
                update_user(existing_user['username'], {
                    "auth_methods.facebook": True,
                    "auth_methods.facebook_id": user_info['id']
                })
        
        # Si toujours pas trouvé, créer un nouvel utilisateur
        if not existing_user:
            username = f"fb_{user_info.get('name', '').replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}"
            new_user = {
                'username': username,
                'email': user_info.get('email', ''),
                'auth_methods': {
                    'facebook': True,
                    'facebook_id': user_info['id']
                },
                'has_face_encoding': False
            }
            insert_user(new_user)
            existing_user = get_user_by_username(username)
        
        # Authentifier l'utilisateur
        st.session_state.authenticated = True
        st.session_state.current_user = existing_user['username']
        st.session_state.auth_method = 'facebook'
        
        return True, existing_user['username']
        
    except Exception as e:
        st.error(f"Erreur lors de l'authentification Facebook: {str(e)}")
        return False, None

# Pages de l'application
def signup_page():
    """Page d'inscription"""
    st.header("Inscription")
    
    signup_tabs = st.tabs(["Email/Mot de passe", "Reconnaissance Faciale", "Google", "Facebook"])
    
    # Onglet Email/Mot de passe
    with signup_tabs[0]:
        with st.form("signup_form_email"):
            username = st.text_input("Nom d'utilisateur")
            email = st.text_input("Email")
            password = st.text_input("Mot de passe", type="password")
            password_confirm = st.text_input("Confirmer le mot de passe", type="password")
            
            if st.form_submit_button("S'inscrire"):
                if not username or not email or not password:
                    st.error("Tous les champs sont obligatoires")
                elif password != password_confirm:
                    st.error("Les mots de passe ne correspondent pas")
                elif get_user_by_username(username):
                    st.error("Ce nom d'utilisateur existe déjà")
                else:
                    # Ajouter l'utilisateur
                    new_user = {
                        'username': username,
                        'password_hash': hash_password(password),
                        'email': email,
                        'auth_methods': {'local': True},
                        'has_face_encoding': False
                    }
                    insert_user(new_user)
                    st.success("Inscription réussie! Vous pouvez maintenant vous connecter.")
    
    # Onglet Reconnaissance Faciale
    with signup_tabs[1]:
        st.write("Inscrivez-vous avec votre visage:")
        
        face_encoding = capture_face()
        
        if face_encoding is not None:
            with st.form("face_signup_form"):
                username = st.text_input("Nom d'utilisateur")
                email = st.text_input("Email")
                
                if st.form_submit_button("S'inscrire avec ce visage"):
                    if not username or not email:
                        st.error("Tous les champs sont obligatoires")
                    elif get_user_by_username(username):
                        st.error("Ce nom d'utilisateur existe déjà")
                    else:
                        # Convertir le tableau numpy en binaire pour MongoDB
                        face_encoding_binary = Binary(pickle.dumps(face_encoding, protocol=2))
                        
                        # Ajouter l'utilisateur avec reconnaissance faciale
                        new_user = {
                            'username': username,
                            'email': email,
                            'auth_methods': {'face': True},
                            'face_encoding': face_encoding_binary,
                            'has_face_encoding': True
                        }
                        insert_user(new_user)
                        st.success("Inscription par reconnaissance faciale réussie!")
    
    # Onglet Google
    with signup_tabs[2]:
        st.write("Inscription avec Google")
        
        if st.button("S'inscrire avec Google"):
            st.session_state.oauth_provider = 'google'
            st.session_state.redirect_page = 'signup'
            auth_url = get_google_auth_url()
            st.markdown(f'<a href="{auth_url}" target="_self">Cliquez ici pour vous connecter avec Google</a>', unsafe_allow_html=True)
    
    # Onglet Facebook
    with signup_tabs[3]:
        st.write("Inscription avec Facebook")
        
        if st.button("S'inscrire avec Facebook"):
            st.session_state.oauth_provider = 'facebook'
            st.session_state.redirect_page = 'signup'
            auth_url = get_facebook_auth_url()
            st.markdown(f'<a href="{auth_url}" target="_self">Cliquez ici pour vous connecter avec Facebook</a>', unsafe_allow_html=True)

def login_page():
    """Page de connexion"""
    st.header("Connexion")
    
    login_tabs = st.tabs(["Email/Mot de passe", "Reconnaissance Faciale", "Google", "Facebook"])
    
    # Onglet Email/Mot de passe
    with login_tabs[0]:
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            
            if st.form_submit_button("Se connecter"):
                if not username or not password:
                    st.error("Veuillez entrer votre nom d'utilisateur et votre mot de passe")
                else:
                    user = get_user_by_username(username)
                    
                    if not user:
                        st.error("Nom d'utilisateur incorrect")
                    elif not user.get('auth_methods', {}).get('local', False):
                        st.error("Ce compte n'utilise pas la connexion par mot de passe")
                    elif user['password_hash'] != hash_password(password):
                        st.error("Mot de passe incorrect")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.session_state.auth_method = 'local'
                        st.success(f"Bienvenue, {username}!")
                        st.rerun()
    
    # Onglet Reconnaissance Faciale
    with login_tabs[1]:
        st.write("Connectez-vous avec votre visage")
        
        face_encoding = capture_face()
        
        if face_encoding is not None:
            users_with_face = get_all_users_with_face()
            best_match = None
            lowest_distance = float('inf')
            threshold = 0.6  # Seuil de similarité
            
            for user in users_with_face:
                if user.get('face_encoding') is not None:
                    # Convertir le binaire en tableau numpy
                    stored_encoding = pickle.loads(user['face_encoding'])
                    distances = face_recognition.face_distance([stored_encoding], face_encoding)
                    distance = distances[0]
                    
                    if distance < lowest_distance:
                        lowest_distance = distance
                        best_match = user
            
            if best_match and lowest_distance < threshold:
                st.session_state.authenticated = True
                st.session_state.current_user = best_match['username']
                st.session_state.auth_method = 'face'
                st.success(f"Visage reconnu! Bienvenue, {best_match['username']}!")
                st.rerun()
            else:
                st.error(f"Aucune correspondance trouvée (distance: {lowest_distance:.2f})")
    
    # Onglet Google
    with login_tabs[2]:
        st.write("Connexion avec Google")
        
        if st.button("Se connecter avec Google"):
            st.session_state.oauth_provider = 'google'
            st.session_state.redirect_page = 'login'
            auth_url = get_google_auth_url()
            st.markdown(f'<a href="{auth_url}" target="_self">Cliquez ici pour vous connecter avec Google</a>', unsafe_allow_html=True)
    
    # Onglet Facebook
    with login_tabs[3]:
        st.write("Connexion avec Facebook")
        
        if st.button("Se connecter avec Facebook"):
            st.session_state.oauth_provider = 'facebook'
            st.session_state.redirect_page = 'login'
            auth_url = get_facebook_auth_url()
            st.markdown(f'<a href="{auth_url}" target="_self">Cliquez ici pour vous connecter avec Facebook</a>', unsafe_allow_html=True)

def profile_page():
    """Page de profil utilisateur"""
    st.header(f"Profil de {st.session_state.current_user}")
    
    user = get_user_by_username(st.session_state.current_user)
    
    # Afficher les informations de base
    st.write(f"Email: {user.get('email')}")
    
    # Afficher et gérer les méthodes d'authentification
    st.subheader("Méthodes d'authentification")
    
    auth_methods = user.get('auth_methods', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Méthodes actives:")
        methods = {
            'local': "Email/Mot de passe",
            'face': "Reconnaissance faciale", 
            'google': "Google",
            'facebook': "Facebook"
        }
        
        for method_id, method_name in methods.items():
            if auth_methods.get(method_id, False) or (method_id == 'face' and user.get('has_face_encoding', False)):
                st.success(f"✅ {method_name}")
            else:
                st.error(f"❌ {method_name}")
    
    with col2:
        st.write("Ajouter des méthodes:")
        
        # Ajouter un mot de passe si non existant
        if not auth_methods.get('local', False):
            with st.expander("Ajouter un mot de passe"):
                with st.form("add_password"):
                    new_password = st.text_input("Nouveau mot de passe", type="password")
                    confirm_password = st.text_input("Confirmer", type="password")
                    
                    if st.form_submit_button("Enregistrer"):
                        if new_password != confirm_password:
                            st.error("Les mots de passe ne correspondent pas")
                        else:
                            update_user(st.session_state.current_user, {
                                "password_hash": hash_password(new_password),
                                "auth_methods.local": True
                            })
                            st.success("Mot de passe ajouté!")
                            st.rerun()
        
        # Ajouter/mettre à jour la reconnaissance faciale
        with st.expander("Reconnaissance faciale"):
            st.write("Mettre à jour votre visage pour l'authentification")
            
            face_encoding = capture_face()
            
            if face_encoding is not None and st.button("Enregistrer ce visage"):
                # Convertir le tableau numpy en binaire pour MongoDB
                face_encoding_binary = Binary(pickle.dumps(face_encoding, protocol=2))
                
                update_user(st.session_state.current_user, {
                    "face_encoding": face_encoding_binary,
                    "has_face_encoding": True,
                    "auth_methods.face": True
                })
                st.success("Visage enregistré!")
                st.rerun()
        
        # Ajouter Google si non existant
        if not auth_methods.get('google', False):
            with st.expander("Connecter Google"):
                if st.button("Lier un compte Google"):
                    st.session_state.oauth_provider = 'google'
                    st.session_state.redirect_page = 'profile'
                    auth_url = get_google_auth_url()
                    st.markdown(f'<a href="{auth_url}" target="_self">Cliquez ici pour lier votre compte Google</a>', unsafe_allow_html=True)
        
        # Ajouter Facebook si non existant
        if not auth_methods.get('facebook', False):
            with st.expander("Connecter Facebook"):
                if st.button("Lier un compte Facebook"):
                    st.session_state.oauth_provider = 'facebook'
                    st.session_state.redirect_page = 'profile'
                    auth_url = get_facebook_auth_url()
                    st.markdown(f'<a href="{auth_url}" target="_self">Cliquez ici pour lier votre compte Facebook</a>', unsafe_allow_html=True)
    
    # Déconnexion
    if st.button("Se déconnecter", type="primary"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.auth_method = None
        st.success("Vous êtes déconnecté")
        st.rerun()

def main():
    """Fonction principale de l'application"""
    st.title("Application CBIR avec Authentification Multiple")
    
    # Vérifier s'il y a un callback OAuth
    if 'code' in st.query_params and 'state' in st.query_params:
        is_authenticated, username = process_oauth_callback()
        if is_authenticated:
            st.success(f"Authentification réussie! Bienvenue, {username}!")
            # Rediriger vers la page principale après une connexion réussie
            st.rerun()
    
    # Barre latérale
    with st.sidebar:
        st.sidebar.title("CBIR App")
        
        if st.session_state.authenticated:
            st.success(f"Connecté en tant que: {st.session_state.current_user}")
            
            # Menu pour utilisateurs connectés
            menu = st.radio("Menu", ["Recherche d'Images", "Mon Profil"])
            
            # Bouton de déconnexion rapide
            if st.button("Déconnexion"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()
        else:
            st.warning("Non connecté")
            menu = st.radio("Menu", ["Accueil", "Connexion", "Inscription"])
    
    # Affichage principal
    if not st.session_state.authenticated:
        if menu == "Connexion":
            login_page()
        elif menu == "Inscription":
            signup_page()
        else:
            # Page d'accueil
            st.header("Bienvenue sur l'Application CBIR")
            
            st.write("""
            Cette application vous permet de rechercher des images basées sur leur contenu 
            visuel plutôt que sur des mots-clés ou des métadonnées. Elle offre également 
            plusieurs méthodes d'authentification pour une expérience personnalisée.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fonctionnalités")
                st.markdown("""
                * Authentification multiple (mot de passe, visage, réseaux sociaux)
                * Recherche d'images par similarité visuelle
                * Interface utilisateur intuitive
                """)
            
            with col2:
                st.subheader("Pour commencer")
                if st.button("Créer un compte"):
                    menu = "Inscription"
                    st.rerun()
                if st.button("Se connecter"):
                    menu = "Connexion"
                    st.rerun()
    else:
        # Pages pour utilisateurs connectés
        if menu == "Recherche d'Images":
            cbir_page()
        elif menu == "Mon Profil":
            profile_page()

# Lancer l'application
if __name__ == "__main__":
    main()