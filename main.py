import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(
    title="Anime Recommendation System API",
    description="Backend API menggunakan Hybrid Filtering (CF + CBF) dengan metadata terintegrasi.",
    version="2.1.0"
)

# ------------------------------------------------------------------------------
# 1. KONFIGURASI CORS
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 2. MEMUAT OBJEK MODEL BINARY (.JOBLIB)
# ------------------------------------------------------------------------------
MODEL_PATH = "data/model"

print(f"[STARTUP] Mencoba memuat model dari: {MODEL_PATH}")

try:
    loaded_model = joblib.load(MODEL_PATH)
    similarity_df = loaded_model['similarity_df']
    genre_similarity_df = loaded_model['genre_similarity_df']
    anime_used = loaded_model['anime_used']
    print("[STARTUP SUCCESS] Seluruh komponen model & metadata berhasil dimuat!")
except Exception as e:
    print(f"[STARTUP ERROR] Gagal memuat file model: {str(e)}")
    similarity_df = pd.DataFrame()
    genre_similarity_df = pd.DataFrame()
    anime_used = pd.DataFrame()

# ------------------------------------------------------------------------------
# 3. SCHEMA MODEL PYDANTIC (Untuk Request POST)
# ------------------------------------------------------------------------------
class FilterRequest(BaseModel):
    genres: Optional[List[str]] = []
    themes: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    top_n: Optional[int] = 20

# ------------------------------------------------------------------------------
# 4. ENDPOINT ROOT & HEALTH CHECK
# ------------------------------------------------------------------------------
@app.get("/")
def root_check():
    return {
        "status": "online",
        "message": "Backend Sistem Rekomendasi Anime Berjalan Lancar!",
        "total_anime_pool": len(anime_used) if not anime_used.empty else 0
    }

# ------------------------------------------------------------------------------
# 5. ENDPOINT SKENARIO A: REKOMENDASI HYBRID (BERDASARKAN JUDUL)
# ------------------------------------------------------------------------------
@app.get("/recommend")
def recommend_anime(title: str, alpha: float = 0.7, top_n: int = 20):
    if anime_used.empty or similarity_df.empty:
        raise HTTPException(status_code=503, detail="Model biner belum siap atau gagal dimuat.")
        
    matched = anime_used[anime_used['title'].str.lower() == title.lower()]
    if matched.empty:
        return {"status": "error", "message": f"Anime '{title}' tidak ditemukan.", "data": []}
        
    anime_id = matched['mal_id'].iloc[0]
    
    if anime_id not in similarity_df.index or anime_id not in genre_similarity_df.index:
        return {"status": "success", "message": "Interaksi data kurang untuk anime acuan ini.", "data": []}
        
    cf_scores = similarity_df[anime_id]
    cbf_scores = genre_similarity_df[anime_id]
    
    common_ids = list(set(cf_scores.index).intersection(set(cbf_scores.index)))
    
    result = pd.DataFrame({'mal_id': common_ids})
    result['cf_score'] = result['mal_id'].map(cf_scores)
    result['cbf_score'] = result['mal_id'].map(cbf_scores)
    
    result = result[result['mal_id'] != anime_id].copy()
    
    if result.empty:
        return {"status": "success", "data": []}
        
    cf_scaler = MinMaxScaler()
    cbf_scaler = MinMaxScaler()
    result['cf_norm'] = cf_scaler.fit_transform(result[['cf_score']])
    result['cbf_norm'] = cbf_scaler.fit_transform(result[['cbf_score']])
    
    result['hybrid_score'] = (alpha * result['cf_norm']) + ((1 - alpha) * result['cbf_norm'])
    
    meta_cols = ['mal_id', 'title', 'image_url', 'score', 'genres', 'synopsis']
    if 'themes' in anime_used.columns:
        meta_cols.append('themes')
        
    result = result.merge(anime_used[meta_cols], on='mal_id')
    result = result.sort_values(by='hybrid_score', ascending=False)
    
    final_list = result.head(top_n).to_dict(orient='records')
    return {"status": "success", "data": final_list}

# ------------------------------------------------------------------------------
# 6. ENDPOINT SKENARIO B: MULTI-TAG FILTERING (GENRE & TEMA)
# ------------------------------------------------------------------------------
def execute_filter_logic(tags_list: List[str], top_n: int = 20):
    """Fungsi internal pengeksekusi filter multi-tag."""
    if anime_used.empty:
        return []
        
    # Standardisasi tag menjadi huruf kecil semua
    cleaned_tags = [t.strip().lower() for t in tags_list if t.strip()]
    
    output_cols = ['mal_id', 'title', 'image_url', 'score', 'genres', 'synopsis']
    if 'themes' in anime_used.columns: 
        output_cols.append('themes')
    
    # JIKA TIDAK ADA FILTER SAMA SEKALI, TAMPILKAN POPULER (FALLBACK)
    if not cleaned_tags:
        return anime_used[output_cols].sort_values(by='score', ascending=False).head(top_n).to_dict(orient='records')
        
    def check_match(row):
        gen_text = str(row['genres']).lower() if 'genres' in row and pd.notna(row['genres']) else ""
        thm_text = str(row['themes']).lower() if 'themes' in row and pd.notna(row['themes']) else ""
        combined_metadata = gen_text + " " + thm_text
        
        # PERBAIKAN SINKRONISASI: Menggunakan 'all' agar wajib lolos semua tag yang dipilih
        return all(tag in combined_metadata for tag in cleaned_tags)
        
    mask = anime_used.apply(check_match, axis=1)
    filtered_df = anime_used[mask].copy()
    
    if filtered_df.empty:
        return []
        
    final_df = filtered_df.sort_values(by='score', ascending=False).head(top_n)
    return final_df[output_cols].to_dict(orient='records')

# Jalur POST: Pembersihan ganda data array JSON
@app.post("/filter")
def filter_anime_post(payload: FilterRequest):
    raw_tags = (payload.genres or []) + (payload.themes or []) + (payload.tags or [])
    all_tags = []
    for item in raw_tags:
        if isinstance(item, str) and item.strip():
            if "," in item:
                all_tags.extend([v.strip() for v in item.split(",") if v.strip()])
            else:
                all_tags.append(item.strip())
                
    filtered_data = execute_filter_logic(tags_list=all_tags, top_n=payload.top_n)
    return {"status": "success", "data": filtered_data}

# Jalur GET Fleksibel: Mampu membaca format genres[], koma, maupun duplikasi URL parameter
@app.get("/filter")
def filter_anime_get(request: Request, top_n: int = 20):
    raw_params = request.query_params
    all_tags = []
    
    for key, value in raw_params.multi_items():
        clean_key = key.replace("[]", "").lower()
        if clean_key in ["genres", "themes", "tags", "genre", "theme", "tag"]:
            if "," in value:
                all_tags.extend([v.strip() for v in value.split(",") if v.strip()])
            else:
                if value.strip():
                    all_tags.append(value.strip())
                    
    filtered_data = execute_filter_logic(tags_list=all_tags, top_n=top_n)
    return {"status": "success", "data": filtered_data}
