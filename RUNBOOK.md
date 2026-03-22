# RUNBOOK.md

## Real-Time Movie Recommendation System

This runbook shows exactly how to start, verify, and demo the full system from scratch.

---

## 1. Project layout

Expected paths on EC2:

```text
~/realtime-rec-system/
  app/api.py
  data/watch_history.csv
  data/movies.csv
  data/reviews.csv
  data/users.csv
  models/train_cf.py
  models/build_candidates.py
  producer/generate_events.py
  streaming/spark_streaming.py
  jars/

~/redis-stable/
~/kafka/
~/spark/
```

---

## 2. One-time dependency check

Run once if needed:

```bash
pip3 install pandas numpy scipy scikit-learn fastapi uvicorn redis kafka-python pyspark pyarrow
```

`pyarrow` is needed because the training script writes parquet.

---

## 3. Offline model training

Run this whenever:

* `watch_history.csv` changes
* `reviews.csv` changes
* `movies.csv` changes
* model logic changes

### Step 3.1 Train review-aware collaborative filtering

```bash
cd ~/realtime-rec-system
python3 models/train_cf.py
```

Expected outputs:

* `models/item_similarity.pkl`
* `models/popular_movies.pkl`
* `models/user_item_interactions.parquet`

### Step 3.2 Build genre-aware user candidates

```bash
cd ~/realtime-rec-system
python3 models/build_candidates.py
```

Expected outputs:

* `models/user_topn.pkl`
* `models/user_topn.json`

---

## 4. Runtime order

Always start services in this order:

1. Redis
2. Kafka
3. Kafka topic check/create
4. Event generator
5. Spark streaming
6. FastAPI

---

## 5. Terminal map

Use separate EC2 terminals.

### Terminal 1

Redis

### Terminal 2

Kafka

### Terminal 3

Kafka topic check + event generator

### Terminal 4

Spark streaming

### Terminal 5

FastAPI

### Terminal 6

Validation commands

---

## 6. Start Redis

### Terminal 1

```bash
cd ~/redis-stable
src/redis-server
```

Leave it running.

### Validation

In another terminal:

```bash
cd ~/redis-stable
src/redis-cli ping
```

Expected:

```bash
PONG
```

---

## 7. Start Kafka

### Terminal 2

Go to Kafka:

```bash
cd ~/kafka
```

Generate a UUID:

```bash
bin/kafka-storage.sh random-uuid
```

Copy the UUID.

Format Kafka storage:

```bash
bin/kafka-storage.sh format -t YOUR_UUID_HERE -c config/kraft/server.properties
```

Start Kafka with low memory settings:

```bash
export KAFKA_HEAP_OPTS="-Xms128m -Xmx256m"
bin/kafka-server-start.sh config/kraft/server.properties
```

Leave it running.

---

## 8. Create or verify Kafka topic

### Terminal 3

```bash
cd ~/kafka
bin/kafka-topics.sh --create --topic user-events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

If it says the topic already exists, that is fine.

Verify:

```bash
bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

Expected:

```bash
user-events
```

---

## 9. Start the movie event generator

### Terminal 3

```bash
cd ~/realtime-rec-system
python3 producer/generate_events.py
```

Leave it running.

Expected output:

```bash
Sent: {'user_id': 45, 'movie_id': 112, 'event_type': 'watch', 'ts': '...', 'genre': 'drama'}
```

---

## 10. Start Spark streaming

### Terminal 4

Clear the old checkpoint first:

```bash
rm -rf /tmp/movie-rec-checkpoint
```

Start Spark:

```bash
cd ~/realtime-rec-system
spark-submit \
  --master local[1] \
  --driver-memory 512m \
  --jars jars/spark-sql-kafka-0-10_2.12-3.5.1.jar,jars/kafka-clients-3.5.1.jar,jars/commons-pool2-2.11.1.jar,jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar \
  streaming/spark_streaming.py
```

Leave it running.

Wait 20 to 40 seconds for micro-batches to process.

---

## 11. Verify live recommendations in Redis

### Terminal 6

Check keys:

```bash
cd ~/redis-stable
src/redis-cli KEYS "rec:user:*"
```

Expected:

```bash
1) "rec:user:67"
2) "rec:user:152"
```

Inspect one user:

```bash
src/redis-cli GET rec:user:67
```

Expected:
JSON recommendation list.

---

## 12. Start FastAPI

### Terminal 5

```bash
cd ~/realtime-rec-system
python3 -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Leave it running.

---

## 13. Validate API on EC2

### Terminal 6

Health:

```bash
curl http://localhost:8000/health
```

Recommendations for a live user:

```bash
curl http://localhost:8000/recommendations/67
```

Use any user ID that exists in Redis.

Possible response sources:

* `redis_live`
* `offline_cf_review_aware_fallback`
* `offline_cf_review_aware_genre_aware`

---

## 14. Validate from your laptop

Open:

```text
http://YOUR_PUBLIC_IP:8000/docs
```

Then test:

```text
http://YOUR_PUBLIC_IP:8000/recommendations/67
```

Use a valid user ID.

---

## 15. How the system behaves

### Offline path

* `train_cf.py` builds the review-aware collaborative filtering model
* `build_candidates.py` builds genre-aware candidate recommendations
* API uses these as fallback recommendations

### Live path

* `generate_events.py` sends movie events to Kafka
* Spark reads Kafka events and scores live activity
* Redis stores per-user near-real-time recommendations
* API prefers Redis live recommendations when available

---

## 16. Restart behavior

If you stop and start the EC2 instance, public IP may change.
Update your SSH command accordingly.

You do not need to retrain the offline model every restart unless data changed.

### Normal restart order

1. Redis
2. Kafka
3. Topic check
4. Event generator
5. Spark
6. FastAPI

---

## 17. Common errors and fixes

### Redis connection refused

Start Redis again:

```bash
cd ~/redis-stable
src/redis-server
```

### Kafka `NoBrokersAvailable`

Kafka is not running. Restart Kafka.

### Kafka `No meta.properties found`

Run:

```bash
cd ~/kafka
bin/kafka-storage.sh random-uuid
bin/kafka-storage.sh format -t YOUR_UUID_HERE -c config/kraft/server.properties
export KAFKA_HEAP_OPTS="-Xms128m -Xmx256m"
bin/kafka-server-start.sh config/kraft/server.properties
```

### Redis has no `rec:user:*` keys

Spark is not processing. Check the Spark terminal for errors.

### API returns 404

Possible reasons:

* user not present in Redis
* offline model not built
* `user_topn.pkl` missing
* user ID not present in training data

### SSH stopped working after EC2 restart

Check:

* instance public IP changed
* security group still allows SSH from your IP

---

## 18. Full command summary

### Offline training

```bash
cd ~/realtime-rec-system
python3 models/train_cf.py
python3 models/build_candidates.py
```

### Redis

```bash
cd ~/redis-stable
src/redis-server
```

### Kafka

```bash
cd ~/kafka
bin/kafka-storage.sh random-uuid
bin/kafka-storage.sh format -t YOUR_UUID_HERE -c config/kraft/server.properties
export KAFKA_HEAP_OPTS="-Xms128m -Xmx256m"
bin/kafka-server-start.sh config/kraft/server.properties
```

### Kafka topic

```bash
cd ~/kafka
bin/kafka-topics.sh --create --topic user-events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### Event generator

```bash
cd ~/realtime-rec-system
python3 producer/generate_events.py
```

### Spark streaming

```bash
rm -rf /tmp/movie-rec-checkpoint
cd ~/realtime-rec-system
spark-submit \
  --master local[1] \
  --driver-memory 512m \
  --jars jars/spark-sql-kafka-0-10_2.12-3.5.1.jar,jars/kafka-clients-3.5.1.jar,jars/commons-pool2-2.11.1.jar,jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar \
  streaming/spark_streaming.py
```

### FastAPI

```bash
cd ~/realtime-rec-system
python3 -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```
