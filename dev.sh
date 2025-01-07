#!/bin/bash

# Function to cleanup background processes on script exit
cleanup() {
    echo "Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Start the backend
echo "Starting backend server..."
cd "$(dirname "$0")"
python3 api.py &

# Start the frontend
echo "Starting frontend..."
cd frontend
npm run dev &

# Wait for all background processes
wait 