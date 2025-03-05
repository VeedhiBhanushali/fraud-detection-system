import redis
import json
import logging

logging.basicConfig(level=logging.INFO)

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_transaction(transaction_id: str):
    try:
        result = cache.get(transaction_id)
        if result:
            logging.info(f"Cache hit for transaction {transaction_id}")
            return json.loads(result)
        logging.info(f"Cache miss for transaction {transaction_id}")
        return None
    except Exception as e:
        logging.error(f"Redis error: {str(e)}")
        return None

def cache_transaction(transaction_id: str, data: dict):
    try:
        cache.set(transaction_id, json.dumps(data), ex=3600)
        logging.info(f"Cached transaction {transaction_id}")
    except Exception as e:
        logging.error(f"Failed to cache transaction {transaction_id}: {str(e)}")
