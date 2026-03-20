import json
import pickle
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import redis
from fastapi import FastAPI, HTTPException

BASE_DIR = Path("/home/ec2-user/realtime-rec-system")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

USER_TOPN_PATH = MODELS_DIR / "user_topn.pkl"
MOVIES_PATH = DATA_DIR / "movies.csv"

app = FastAPI(title="Movie Recommendation API")

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

USER_TOPN = {}
MOVIE_METADATA: Dict[str, Dict[str, Any]] = {}


def load_user_topn():
    global USER_TOPN
    if USER_TOPN_PATH.exists():
        with open(USER_TOPN_PATH, "rb") as f:
            USER_TOPN = pickle.load(f)
        print(f"Loaded offline CF candidates for {len(USER_TOPN):,} users")
    else:
        print(f"Offline CF file not found at {USER_TOPN_PATH}")


def load_movie_metadata():
    global MOVIE_METADATA

    if not MOVIES_PATH.exists():
        print(f"Movie metadata file not found at {MOVIES_PATH}")
        return

    df = pd.read_csv(MOVIES_PATH)

    required = {"movie_id", "title", "genre_primary", "genre_secondary", "content_type", "language", "release_year", "imdb_rating"}
    existing = [c for c in required if c in df.columns]

    df = df[existing].copy()
    df["movie_id"] = df["movie_id"].astype(str)

    MOVIE_METADATA = {
        row["movie_id"]: {
            k: row[k]
            for k in existing
            if k != "movie_id" and pd.notna(row[k])
        }
        for _, row in df.iterrows()
    }

    print(f"Loaded movie metadata for {len(MOVIE_METADATA):,} movies")


def enrich_recommendations(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []

    for i, rec in enumerate(recs, start=1):
        movie_id = str(rec.get("movie_id"))

        item = {
            "movie_id": movie_id,
            "score": rec.get("score", rec.get("cf_score", 0.0)),
            "rank": rec.get("rank", i),
            "source": rec.get("source", "unknown"),
        }

        meta = MOVIE_METADATA.get(movie_id, {})
        item.update(meta)

        enriched.append(item)

    return enriched


@app.on_event("startup")
def startup_event():
    load_user_topn()
    load_movie_metadata()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "offline_cf_loaded": bool(USER_TOPN),
        "movie_metadata_loaded": bool(MOVIE_METADATA),
    }


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):
    user_key = str(user_id)

    # 1. Prefer live Redis recommendations
    redis_data = redis_client.get(f"rec:user:{user_key}")
    if redis_data:
        live_recs = json.loads(redis_data)

        for i, rec in enumerate(live_recs, start=1):
            rec.setdefault("rank", i)
            rec.setdefault("source", "live_streaming")

        return {
            "user_id": user_id,
            "source": "redis_live",
            "recommendations": enrich_recommendations(live_recs),
        }

    # 2. Fall back to offline CF
    offline_recs = USER_TOPN.get(user_key)
    if offline_recs:
        payload = []
        for i, rec in enumerate(offline_recs[:10], start=1):
            payload.append({
                "movie_id": str(rec["movie_id"]),
                "cf_score": float(rec.get("cf_score", 0.0)),
                "genre_boost": float(rec.get("genre_boost", 0.0)),
                "score": float(rec.get("final_score", rec.get("cf_score", 0.0))),
                "rank": i,
                "source": "offline_cf_review_aware_genre_aware",
            })

        return {
            "user_id": user_id,
            "source": "offline_cf_fallback",
            "recommendations": enrich_recommendations(payload),
        }

    raise HTTPException(
        status_code=404,
        detail=f"No live or offline recommendations found for user {user_id}",
    )