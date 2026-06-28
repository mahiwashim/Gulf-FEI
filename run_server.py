#!/usr/bin/env python
"""
Gulf FEI RAG System - Server Startup Script
This script ensures all dependencies are loaded and starts the FastAPI server.
"""

import sys
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env():
    """Check if .env file exists and has GROQ_API_KEY"""
    if not os.path.exists('.env'):
        logger.error("❌ .env file not found!")
        logger.info("Please create a .env file with your GROQ_API_KEY")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not set in .env file!")
        return False
    
    logger.info("✓ .env configuration OK")
    return True

def check_vector_db():
    """Check if vector database exists"""
    if os.path.exists("vector_db/faiss_index.idx") or os.path.exists("vector_db/index.faiss"):
        logger.info("✓ Vector database found")
    else:
        logger.warning("⚠ Vector database not found. It will be loaded on startup.")

def check_static_dir():
    """Create static directory if it doesn't exist"""
    if not os.path.exists("static"):
        os.makedirs("static")
        logger.info("✓ Created static directory")
    else:
        logger.info("✓ Static directory exists")

def check_templates_dir():
    """Check if templates directory exists"""
    if not os.path.exists("templates"):
        logger.error("❌ Templates directory not found!")
        return False
    logger.info("✓ Templates directory exists")
    return True

def main():
    logger.info("=" * 50)
    logger.info("Gulf FEI RAG System - Server Startup")
    logger.info("=" * 50)
    
    # Pre-flight checks
    logger.info("\n📋 Running pre-flight checks...")
    
    if not check_env():
        sys.exit(1)
    
    if not check_templates_dir():
        sys.exit(1)
    
    check_vector_db()
    check_static_dir()
    
    # Start server
    logger.info("\n" + "=" * 50)
    logger.info("🚀 Starting FastAPI server...")
    logger.info("=" * 50)
    logger.info("📍 Server will be available at: http://localhost:8000")
    logger.info("🔧 API endpoint: http://localhost:8000/query")
    logger.info("📊 Health check: http://localhost:8000/health")
    logger.info("\n💡 Tip: Open http://localhost:8000 in your browser")
    logger.info("=" * 50 + "\n")
    
    # Start uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

if __name__ == "__main__":
    main()
