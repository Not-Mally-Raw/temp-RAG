#!/usr/bin/env python3
"""
Automated Code Fixer for RAG System
Fixes import statements and other code issues automatically
"""

import re
import os
from pathlib import Path

def fix_file(filepath, fixes):
    """Apply fixes to a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_text, new_text in fixes:
            content = content.replace(old_text, new_text)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"   ❌ Error fixing {filepath}: {e}")
        return False

def main():
    print("🔧 Automated Code Fixer for RAG System")
    print("=" * 50)
    
    base_path = Path("/Users/spandankewte/RAG-System")
    fixes_applied = 0
    
    # Fix 1: core/universal_rag_system.py
    print("\n1. Fixing core/universal_rag_system.py...")
    file1 = base_path / "core" / "universal_rag_system.py"
    fixes1 = [
        (
            "from implicit_rule_extractor import ImplicitRuleExtractor, ImplicitRule",
            "from core.implicit_rule_extractor import ImplicitRuleExtractor, ImplicitRule"
        ),
        (
            "from enhanced_rag_db import",
            "from core.enhanced_rag_db import"
        ),
        (
            "from langchain.vectorstores import Chroma",
            "from langchain_chroma import Chroma"
        ),
        (
            "from langchain.embeddings.base import Embeddings",
            "from langchain_core.embeddings import Embeddings"
        ),
    ]
    if fix_file(file1, fixes1):
        print("   ✅ Fixed import statements")
        fixes_applied += 1
    else:
        print("   ℹ️  No changes needed or file not found")
    
    # Fix 2: core/enhanced_rag_db.py
    print("\n2. Fixing core/enhanced_rag_db.py...")
    file2 = base_path / "core" / "enhanced_rag_db.py"
    fixes2 = [
        (
            "from langchain.vectorstores import Chroma",
            "from langchain_chroma import Chroma"
        ),
        (
            "from langchain.embeddings.base import Embeddings",
            "from langchain_core.embeddings import Embeddings"
        ),
        (
            "from langchain.text_splitter import RecursiveCharacterTextSplitter",
            "from langchain_text_splitters import RecursiveCharacterTextSplitter"
        ),
        (
            "from langchain.docstore.document import Document",
            "from langchain_core.documents import Document"
        ),
        (
            "from langchain.schema import BaseRetriever",
            "from langchain_core.retrievers import BaseRetriever"
        ),
    ]
    if fix_file(file2, fixes2):
        print("   ✅ Fixed LangChain imports")
        fixes_applied += 1
    else:
        print("   ℹ️  No changes needed or file not found")
    
    # Fix 3: core/enhanced_universal_classifier.py - Add missing os import
    print("\n3. Fixing core/enhanced_universal_classifier.py...")
    file3 = base_path / "core" / "enhanced_universal_classifier.py"
    try:
        with open(file3, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if os is already imported
        if 'import os' not in content:
            # Add import after other standard library imports
            lines = content.split('\n')
            insert_pos = 0
            
            # Find the right place to insert (after initial comments/docstrings)
            in_docstring = False
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                elif not in_docstring and (line.startswith('import ') or line.startswith('from ')):
                    insert_pos = i
                    break
            
            # Insert import os before the first import
            if insert_pos > 0:
                lines.insert(insert_pos, 'import os')
                content = '\n'.join(lines)
                
                with open(file3, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("   ✅ Added 'import os'")
                fixes_applied += 1
            else:
                print("   ⚠️  Could not find insertion point")
        else:
            print("   ℹ️  'import os' already present")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Fix 4: core/implicit_rule_extractor.py - Update NLTK downloads
    print("\n4. Fixing core/implicit_rule_extractor.py...")
    file4 = base_path / "core" / "implicit_rule_extractor.py"
    fixes4 = [
        (
            "nltk.data.find('tokenizers/punkt')",
            "nltk.data.find('tokenizers/punkt_tab')"
        ),
        (
            "nltk.download('punkt')",
            "nltk.download('punkt_tab')"
        ),
        (
            "nltk.data.find('taggers/averaged_perceptron_tagger')",
            "nltk.data.find('taggers/averaged_perceptron_tagger_eng')"
        ),
        (
            "nltk.download('averaged_perceptron_tagger')",
            "nltk.download('averaged_perceptron_tagger_eng')"
        ),
        (
            "nltk.data.find('chunkers/maxent_ne_chunker')",
            "nltk.data.find('chunkers/maxent_ne_chunker_tab')"
        ),
        (
            "nltk.download('maxent_ne_chunker')",
            "nltk.download('maxent_ne_chunker_tab')"
        ),
    ]
    if fix_file(file4, fixes4):
        print("   ✅ Updated NLTK resource names")
        fixes_applied += 1
    else:
        print("   ℹ️  No changes needed or file not found")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"✅ Applied fixes to {fixes_applied} files")
    print("\n🧪 Next Steps:")
    print("1. Run: python quick_test.py")
    print("2. If all tests pass, run: streamlit run main_app.py")
    print("\n📋 For detailed error analysis, see:")
    print("   ERROR_ANALYSIS_AND_SOLUTIONS.md")

if __name__ == "__main__":
    main()
