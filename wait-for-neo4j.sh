#!/bin/bash
set -e

echo "Waiting for Neo4j..."
max_attempts=20
attempt=0

# OrbStack: Faster connection check
while [ $attempt -lt $max_attempts ]; do
    if exec 3<>/dev/tcp/neo4j/7687 2>/dev/null; then
        exec 3>&-
        echo "Neo4j is ready!"
        break
    fi
    echo "  Attempt $((attempt+1))/$max_attempts"
    sleep 1
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "Neo4j connection timeout"
    exit 1
fi

# Brief stabilization
sleep 2
echo "Starting tests..."
exec "$@"