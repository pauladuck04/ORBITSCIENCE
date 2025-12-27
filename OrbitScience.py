import os
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

app = Flask(__name__)

# --- CARGA Y CONFIGURACIÓN ---
url = 'https://raw.githubusercontent.com/qpabloquiroga/ABP/refs/heads/main/arxiv.json'
df = pd.read_json(url, lines=True)
df = df.dropna(subset=['abstract'])

df['title'] = df['title'].str.strip() # Esto quita los saltos de linea que rompen las URLs

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess_abstract(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]','', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['abstract_processed'] = df['abstract'].apply(preprocess_abstract)
vectorizer = TfidfVectorizer(max_features=3000)
tfidf_matrix = vectorizer.fit_transform(df['abstract_processed'])

# --- TUS FUNCIONES DE FEEDBACK (Revisadas) ---
FEEDBACK_FILE = "user_feedback.json"

def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump({"likes": [], "dislikes": []}, f, indent=4)
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)

def registrar_feedback(title, action):
    data = load_feedback()
    if title not in df['title'].values:
        return "Paper no encontrado."
    
    # Limpieza de duplicados
    data["likes"] = [t for t in data["likes"] if t != title]
    data["dislikes"] = [t for t in data["dislikes"] if t != title]

    if action == "like":
        data["likes"].append(title)
    elif action == "dislike":
        data["dislikes"].append(title)
    
    save_feedback(data)
    return f"Registrado: {action} en {title}"

# --- FUNCIÓN DE PREDICCIÓN ---
@app.route('/predecir/<path:title>')
def obtener_prediccion_ia(title):
    title = title.strip()
    data = load_feedback()
    
    if len(data["likes"]) < 1 and len(data["dislikes"]) < 1:
        return {"prediccion": "Vota para activar OrbitAI", "color": "#aaa"}

    try:
        row = df[df['title'] == title]
        if row.empty:
            return {"prediccion": "Paper no encontrado", "color": "orange"}
            
        abstract_actual = row['abstract'].iloc[0]
        vec_actual = vectorizer.transform([abstract_actual])

        def calcular_proximidad(lista_titulos):
            if not lista_titulos: return 0
            abstracts = df[df['title'].isin(lista_titulos)]['abstract']
            if abstracts.empty: return 0
            vectores = vectorizer.transform(abstracts)
            return cosine_similarity(vec_actual, vectores).mean()

        sim_like = calcular_proximidad(data["likes"])
        sim_dislike = calcular_proximidad(data["dislikes"])

        # --- NUEVA LÓGICA CON UMBRAL DEL 15% ---
        umbral = 0.15
        porcentaje = int(sim_like * 100)

        # Caso 1: Alta coincidencia (Verde)
        if sim_like >= umbral and sim_like >= sim_dislike:
            return {
                "prediccion": f"OrbitAI: {porcentaje}% de coincidencia. Te gustara!",
                "color": "#28a745"
            }
        # Caso 2: Coincidencia moderada o baja (Amarillo/Naranja)
        elif sim_like > 0 and sim_like < umbral:
            return {
                "prediccion": f"OrbitAI: {porcentaje}% de coincidencia. Interes bajo.",
                "color": "#ffc107"
            }
        # Caso 3: No coincide o predomina el dislike (Rojo)
        else:
            return {
                "prediccion": "OrbitAI: No es tu estilo.",
                "color": "#dc3545"
            }
            
    except Exception as e:
        return {"prediccion": "Error al analizar similitudes", "color": "yellow"}

# --- RUTAS ---
# Ruta paginada (mejora UX)
@app.route('/')
def home():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    start = (page - 1) * per_page
    papers_list = df.iloc[start:start+per_page].to_dict('records')
    return render_template('index.html', papers=papers_list, page=page)

@app.route('/predecir', methods=['POST'])
def api_predecir():
    data_json = request.get_json()
    title = data_json.get('title', '').strip()
    return jsonify(obtener_prediccion_ia(title))

@app.route('/feedback/<action>', methods=['POST'])
def api_procesar_voto(action):
    data_json = request.get_json()
    title = data_json.get('title')
    
    if not title or action not in ['like', 'dislike']:
        return jsonify({'error': 'Datos invalidos'}), 400
    
    registrar_feedback(title.strip(), action)
    
    return jsonify({
        'status': 'success', 
        'prediction': obtener_prediccion_ia(title)
    })

@app.route('/similares', methods=['POST'])
def api_similares():
    try:
        data = request.get_json()
        title = data.get('title', '').strip()
        
        # Localizar el paper en el DataFrame
        row = df[df['title'] == title]
        if row.empty: return jsonify([])
        
        idx = row.index[0]
        # Calcular similitud contra todos
        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Los 3 más parecidos (saltando el 1ero)
        indices = sims.argsort()[-4:-1][::-1]
        
        res = []
        for i in indices:
            res.append({
                "title": df.loc[i, 'title'],
                "porcentaje": int(sims[i] * 100)
            })
        return jsonify(res)
    except:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)