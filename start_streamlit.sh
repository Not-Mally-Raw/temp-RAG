#!/bin/bash
# Start Streamlit Application for RAG System

echo "========================================"
echo "Starting RAG System Streamlit Application"
echo "========================================"

cd /workspace

echo ""
echo "Starting Streamlit on port 8501..."
echo ""

# Start streamlit in the background
streamlit run main_app.py \
    --server.headless true \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

echo ""
echo "========================================"
echo "Application running!"
echo "========================================"
