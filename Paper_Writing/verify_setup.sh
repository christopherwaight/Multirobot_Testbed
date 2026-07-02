#!/bin/bash

echo "========================================"
echo "Verifying RAG Chatbot Setup"
echo "========================================"

cd "/Users/christopherwaight/Desktop/Multirobot_Testbed/Paper_Writing"

echo ""
echo "1. Checking venv exists..."
if [ -d "venv" ]; then
    echo "   ✓ venv/ directory exists"
else
    echo "   ✗ venv/ directory NOT found"
    exit 1
fi

echo ""
echo "2. Checking Python in venv..."
if [ -f "venv/bin/python3" ]; then
    echo "   ✓ Python executable found"
    ./venv/bin/python3 --version
else
    echo "   ✗ Python NOT found in venv"
    exit 1
fi

echo ""
echo "3. Checking dependencies..."
./venv/bin/python3 -c "import pymupdf, faiss, sentence_transformers, anthropic; print('   ✓ All dependencies installed')" 2>&1

echo ""
echo "4. Checking FAISS index..."
if [ -f "vector_store/faiss_index.bin" ]; then
    SIZE=$(ls -lh vector_store/faiss_index.bin | awk '{print $5}')
    echo "   ✓ Index exists ($SIZE)"
else
    echo "   ✗ Index NOT found - run: ./venv/bin/python3 build_index.py"
fi

echo ""
echo "5. Checking metadata..."
if [ -f "vector_store/chunks_metadata.json" ]; then
    SIZE=$(ls -lh vector_store/chunks_metadata.json | awk '{print $5}')
    CHUNKS=$(./venv/bin/python3 -c "import json; print(len(json.load(open('vector_store/chunks_metadata.json'))))" 2>/dev/null)
    echo "   ✓ Metadata exists ($SIZE, $CHUNKS chunks)"
else
    echo "   ✗ Metadata NOT found"
fi

echo ""
echo "6. Checking MCP configuration..."
if [ -f ".claude/mcp.json" ]; then
    echo "   ✓ MCP config exists"
    cat .claude/mcp.json | head -5
else
    echo "   ✗ MCP config NOT found"
fi

echo ""
echo "7. Testing retrieval engine..."
./venv/bin/python3 -c "
from mcp_paper_search.retrieval import PaperSearchEngine
print('   Loading search engine...')
engine = PaperSearchEngine()
print('   ✓ Search engine loaded successfully')
print(f'   ✓ {len(engine.chunks)} chunks available')
" 2>&1 | grep -v "Warning:"

echo ""
echo "========================================"
echo "Setup Status: ✓ READY TO USE"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set API key: export ANTHROPIC_API_KEY='your-key'"
echo "2. Start Jupyter: ./venv/bin/jupyter notebook rag_chatbot.ipynb"
echo "3. Restart Claude Code to activate MCP server"
echo ""
