#!/bin/bash

echo "🔧 RAG System - Automated Fix Script"
echo "===================================="
echo ""

# Change to RAG-System directory
cd /Users/spandankewte/RAG-System

echo "📦 Phase 1: Installing Missing Resources"
echo "----------------------------------------"

echo "1. Installing NLTK data resources..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('maxent_ne_chunker_tab', quiet=True); nltk.download('words', quiet=True); print('   ✅ NLTK resources installed')"

echo "2. Installing spaCy English model..."
python -m spacy download en_core_web_sm --quiet 2>/dev/null && echo "   ✅ spaCy model installed" || echo "   ⚠️  spaCy model installation failed (may already be installed)"

echo ""
echo "✅ Phase 1 Complete - Resources Installed"
echo ""
echo "⚠️  Phase 2: Manual Code Fixes Required"
echo "----------------------------------------"
echo "The following files need manual editing:"
echo ""
echo "1. core/universal_rag_system.py (line 26)"
echo "   Change: from implicit_rule_extractor import ImplicitRuleExtractor"
echo "   To:     from core.implicit_rule_extractor import ImplicitRuleExtractor"
echo ""
echo "2. core/enhanced_rag_db.py"
echo "   - Lines 18-23: Update LangChain imports"
echo "   - Line 26: Add 'core.' prefix to implicit_rule_extractor import"
echo ""
echo "3. core/enhanced_universal_classifier.py"
echo "   - Add 'import os' at the top"
echo ""
echo "4. core/implicit_rule_extractor.py (lines 20-31)"
echo "   - Update NLTK downloads to use new resource names"
echo ""
echo "📋 See ERROR_ANALYSIS_AND_SOLUTIONS.md for detailed instructions"
echo ""
echo "🧪 After manual fixes, run: python quick_test.py"
echo "🚀 If tests pass, run: streamlit run main_app.py"
