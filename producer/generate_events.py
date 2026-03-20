import json
import time
import random
from datetime import datetime, timezone
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

event_types = ["view", "click", "watchlist", "watch"]
genres = [
    "action",
    "comedy",
    "drama",
    "thriller",
    "sci-fi",
    "romance",
    "horror",
    "documentary",
    "animation",
]

def generate_event():
    return {
        "user_id": random.randint(1, 200),
        "movie_id": random.randint(1, 1000),
        "event_type": random.choices(
            event_types, weights=[0.5, 0.2, 0.15, 0.15], k=1
        )[0],
        "ts": datetime.now(timezone.utc).isoformat(),
        "genre": random.choice(genres),
    }

while True:
    event = generate_event()
    producer.send("user-events", event)
    producer.flush()
    print("Sent:", event, flush=True)
    time.sleep(random.uniform(0.5, 1.5))