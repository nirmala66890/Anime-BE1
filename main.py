import os
import joblib
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------------------------
# VARIABEL GLOBAL
# ------------------------------------------------------------------------------
similarity_df = pd.DataFrame()
genre_similarity_df = pd.DataFrame()
anime_used = pd.DataFrame()

# ------------------------------------------------------------------------------
# LIFESPAN: DOWNLOAD & LOAD MODEL
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global similarity_df, genre_similarity_df, anime_used
    
    print("[STARTUP] Menyiapkan model...", flush=True)
    try:
        hf_token = os.getenv("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id="Itsmeh/BE1",
            filename="model",
            repo_type="model",
            token=hf_token
        )
        loaded_model = joblib.load(model_path)
        similarity_df = loaded_model['similarity_df']
        genre_similarity_df = loaded_model['genre_similarity_df']
        anime_used = loaded_model['anime_used']
        print("[STARTUP SUCCESS] Model dimuat.", flush=True)
    except Exception as e:
        print(f"[STARTUP CRASH] Error: {str(e)}", flush=True)
    yield
    print("[SHUTDOWN] Mematikan aplikasi.", flush=True)

# ------------------------------------------------------------------------------
# INISIALISASI FASTAPI
# ------------------------------------------------------------------------------
app = FastAPI(title="Anime API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/")
def root_check():
    return {"status": "online", "total_anime": len(anime_used)}

@app.get("/recommend")
def recommend_anime(title: str, alpha: float = 0.7, top_n: int = 20):
    if anime_used.empty:
        raise HTTPException(status_code=503, detail="Model belum siap.")
        
    matched = anime_used[anime_used['title'].str.lower() == title.lower()]
    if matched.empty:
        return {"status": "error", "message": "Anime tidak ditemukan.", "data": []}
        
    anime_id = matched['mal_id'].iloc[0]
    
    # Perhitungan hybrid
    cf_scores = similarity_df[anime_id]
    cbf_scores = genre_similarity_df[anime_id]
    common_ids = list(set(cf_scores.index).intersection(set(cbf_scores.index)))
    
    result = pd.DataFrame({'mal_id': common_ids})
    result['cf_score'] = result['mal_id'].map(cf_scores)
    result['cbf_score'] = result['mal_id'].map(cbf_scores)
    result = result[result['mal_id'] != anime_id].copy()
    
    cf_scaler = MinMaxScaler()
    cbf_scaler = MinMaxScaler()
    result['cf_norm'] = cf_scaler.fit_transform(result[['cf_score']])
    result['cbf_norm'] = cbf_scaler.fit_transform(result[['cbf_score']])
    result['hybrid_score'] = (alpha * result['cf_norm']) + ((1 - alpha) * result['cbf_norm'])
    
    result = result.merge(anime_used[['mal_id', 'title', 'image_url', 'score', 'genres', 'synopsis']], on='mal_id')
    result = result.sort_values(by='hybrid_score', ascending=False)
    
    # Sanitasi
    result = result.replace([np.inf, -np.inf], np.nan).fillna("")
    return {"status": "success", "data": result.head(top_n).to_dict(orient='records')}

@app.get("/filter")
def filter_anime_get(request: Request, top_n: int = 20):
    raw_params = request.query_params
    all_tags = []
    for key, value in raw_params.multi_items():
        if key.lower() in ["genres", "themes", "tags", "genre", "theme", "tag"]:
            if "," in value: all_tags.extend([v.strip() for v in value.split(",") if v.strip()])
            else: all_tags.append(value.strip())
            
    if anime_used.empty: return {"status": "success", "data": []}
    
    def check_match(row):
        gen_text = str(row['genres']).lower() if 'genres' in row else ""
        thm_text = str(row['themes']).lower() if 'themes' in row else ""
        return all(tag.lower() in (gen_text + thm_text) for tag in all_tags)
        
    mask = anime_used.apply(check_match, axis=1)
    filtered_df = anime_used[mask].sort_values(by='score', ascending=False).head(top_n)
    
    # Sanitasi
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).fillna("")
    return {"status": "success", "data": filtered_df.to_dict(orient='records')}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
