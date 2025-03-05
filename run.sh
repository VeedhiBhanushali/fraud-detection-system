#!/bin/bash

echo "🚀 Starting Fraud Detection System..."

# Set port variables
FASTAPI_PORT=8001
DOCKER_PORT=8000
KAFKA_PORT=9092
ZOOKEEPER_PORT=2181
REDIS_PORT=6379

# Function to check if a port is in use
check_port() {
    if nc -z localhost "$1"; then
        echo "❌ Port $1 is already in use. Please free it or modify the script."
        return 1
    else
        return 0
    fi
}

# Function to kill a process using a port
kill_port_process() {
    local PORT=$1
    local PID=$(lsof -t -i:$PORT)
    if [ ! -z "$PID" ]; then
        echo "🛑 Killing process $PID using port $PORT..."
        kill -9 "$PID"
        sleep 3
    fi
}

# Function to wait for a service to start
wait_for_service() {
    local PORT=$1
    local NAME=$2
    echo "⏳ Waiting for $NAME to start on port $PORT..."
    while ! nc -z localhost "$PORT"; do
        sleep 2
    done
    echo "✅ $NAME is running on port $PORT."
}

# Activate Virtual Environment
echo "🔹 Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Start Redis
if pgrep -x "redis-server" > /dev/null; then
    echo "✅ Redis is already running."
else
    echo "🔹 Starting Redis..."
    redis-server --daemonize yes
    wait_for_service $REDIS_PORT "Redis"
fi

# Start Zookeeper
if nc -z localhost $ZOOKEEPER_PORT; then
    echo "✅ Zookeeper is already running."
else
    echo "🔹 Starting Zookeeper..."
    /opt/homebrew/Cellar/kafka/3.9.0/bin/zookeeper-server-start -daemon /opt/homebrew/etc/kafka/zookeeper.properties
    wait_for_service $ZOOKEEPER_PORT "Zookeeper"
fi

# Start Kafka
if nc -z localhost $KAFKA_PORT; then
    echo "✅ Kafka is already running."
else
    echo "🛠 Removing old Kafka lock file..."
    rm -rf /opt/homebrew/var/lib/kafka-logs/.lock

    echo "🔹 Starting Kafka Broker..."
    /opt/homebrew/Cellar/kafka/3.9.0/bin/kafka-server-start -daemon /opt/homebrew/etc/kafka/server.properties
    wait_for_service $KAFKA_PORT "Kafka"
fi

# Ensure Docker is Running
if ! docker info > /dev/null 2>&1; then
    echo "🔹 Docker is not running. Starting Docker..."
    open -a Docker
    sleep 10
fi

# Re-check Docker status
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is still not running. Exiting script."
    exit 1
fi

# Kill any existing FastAPI process using the port
kill_port_process $FASTAPI_PORT

# Start FastAPI
check_port $FASTAPI_PORT && {
    echo "🔹 Starting FastAPI API on port $FASTAPI_PORT..."
    uvicorn src.fastapi_service:app --host 0.0.0.0 --port $FASTAPI_PORT &
    wait_for_service $FASTAPI_PORT "FastAPI"
}

# Start Kafka Producer
echo "🔹 Starting Kafka Producer..."
python src/kafka_producer.py &

# Start Kafka Consumer
echo "🔹 Starting Kafka Consumer..."
python src/kafka_consumer.py &

# Run Docker Container (Ensure it's running first)
if [ "$(docker ps -q -f name=fraud-detection)" ]; then
    echo "✅ Docker container is already running."
else
    echo "🔹 Starting Docker Container on port $DOCKER_PORT..."
    docker run -p 8001:8001 fraud-detection &


fi

echo "✅ Fraud Detection System is running!"

# Trap to ensure cleanup on exit
trap "echo '🛑 Stopping all services...'; kill -- -$$" SIGINT SIGTERM EXIT

# Keep script alive
wait
