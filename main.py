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
    
    print("[STARTUP] Menyiapkan model dari Hugging Face Hub...", flush=True)
    
    try:
        hf_token = os.getenv("HF_TOKEN")
        print(f"[STARTUP] Mengunduh/Memuat file model dari Itsmeh/BE1...", flush=True)
        
        model_path = hf_hub_download(
            repo_id="Itsmeh/BE1",
            filename="model",
            repo_type="model",
            token=hf_token
        )
        
        print(f"[STARTUP] Model berhasil di-load dari: {model_path}", flush=True)
        loaded_model = joblib.load(model_path)
        
        similarity_df = loaded_model['similarity_df']
        genre_similarity_df = loaded_model['genre_similarity_df']
        anime_used = loaded_model['anime_used']
        
        print(f"[STARTUP SUCCESS] Total pool: {len(anime_used)} anime.", flush=True)
        
    except Exception as e:
        print(f"[STARTUP CRASH] Gagal mengunduh/memuat model: {str(e)}", flush=True)
            
    yield
    print("[SHUTDOWN] Mematikan aplikasi.", flush=True)

# ------------------------------------------------------------------------------
# INISIALISASI FASTAPI
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Anime Recommendation System API",
    description="Backend API Hybrid Filtering (CF + CBF) - Railway Ready",
    version="3.0.0",
    lifespan=lifespan
)

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
    return {
        "status": "online",
        "total_anime_pool": len(anime_used) if not anime_used.empty else 0
    }

@app.get("/recommend")
def recommend_anime(title: str, alpha: float = 0.7, top_n: int = 20):
    if anime_used.empty or similarity_df.empty:
        raise HTTPException(status_code=503, detail="Model belum siap.")
        
    matched = anime_used[anime_used['title'].str.lower() == title.lower()]
    if matched.empty:
        return {"status": "error", "message": f"Anime '{title}' tidak ditemukan.", "data": []}
        
    anime_id = matched['mal_id'].iloc[0]
    
    if anime_id not in similarity_df.index or anime_id not in genre_similarity_df.index:
        return {"status": "success", "message": "Data tidak cukup.", "data": []}
        
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
    
    meta_cols = ['mal_id', 'title', 'image_url', 'score', 'genres', 'synopsis']
    if 'themes' in anime_used.columns: meta_cols.append('themes')
        
    result = result.merge(anime_used[meta_cols], on='mal_id')
    result = result.sort_values(by='hybrid_score', ascending=False)
    
    # Sanitasi data sebelum dikirim agar tidak error JSON
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
    
    # PERBAIKAN: Sanitasi data agar tidak ada NaN atau Infinity
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
    filtered_df = filtered_df.fillna("")
    
    return {"status": "success", "data": filtered_df.to_dict(orient='records')}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
    result['cf_score'] = result['mal_id'].map(cf_scores)
    result['cbf_score'] = result['mal_id'].map(cbf_scores)
    
    result = result[result['mal_id'] != anime_id].copy()
    
    cf_scaler = MinMaxScaler()
    cbf_scaler = MinMaxScaler()
    result['cf_norm'] = cf_scaler.fit_transform(result[['cf_score']])
    result['cbf_norm'] = cbf_scaler.fit_transform(result[['cbf_score']])
    
    result['hybrid_score'] = (alpha * result['cf_norm']) + ((1 - alpha) * result['cbf_norm'])
    
    meta_cols = ['mal_id', 'title', 'image_url', 'score', 'genres', 'synopsis']
    if 'themes' in anime_used.columns: meta_cols.append('themes')
        
    result = result.merge(anime_used[meta_cols], on='mal_id')
    result = result.sort_values(by='hybrid_score', ascending=False)
    
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
    
    return {"status": "success", "data": filtered_df.to_dict(orient='records')}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
