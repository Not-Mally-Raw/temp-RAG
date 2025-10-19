# Contributing to Enhanced RAG System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/temp-RAG.git
   cd temp-RAG
   ```
3. **Set up your environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Run tests** to ensure everything works:
   ```bash
   python tests/test_dfm_pipeline.py
   ```

## 📝 Code Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

### Example Function

```python
def extract_rules_from_text(
    text: str, 
    confidence_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Extract manufacturing rules from text content.
    
    Args:
        text: Input text to process
        confidence_threshold: Minimum confidence score for rules
        
    Returns:
        List of extracted rule dictionaries with type, content, and confidence
        
    Raises:
        ValueError: If text is empty or threshold is invalid
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    # Implementation...
    return rules
```

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Local imports
from core.enhanced_rag_db import EnhancedManufacturingRAG
from config import RAGConfig
```

## 🏗️ Project Structure

```
temp-RAG/
├── core/              # Core RAG and DFM pipeline modules
├── extractors/        # Document extraction (text, tables, images)
├── generators/        # Content and feature generators
├── pages/             # Streamlit UI pages
├── tests/             # Test suite
├── data/              # Sample data and documents
├── docs/              # Documentation
└── config.py          # System configuration
```

### Where to Add New Features

- **New extraction methods**: `extractors/`
- **Pipeline improvements**: `core/dfm_pipeline.py` or `core/enhanced_rag_db.py`
- **UI features**: `pages/`
- **Configuration options**: `config.py`
- **Tests**: `tests/`

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_dfm_pipeline.py

# Run with coverage
pytest --cov=core --cov=extractors tests/
```

### Writing Tests

```python
def test_feature_name():
    """Test description of what is being tested."""
    # Arrange
    input_data = "test data"
    expected_output = "expected result"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output, "Error message"
```

### Test Requirements

- All new features must include tests
- Tests should be independent (no dependencies between tests)
- Use meaningful test names that describe what's being tested
- Include edge cases and error conditions

## 📚 Documentation

### Adding Documentation

When adding new features:

1. Update relevant `.md` files in `docs/`
2. Add docstrings to all public functions
3. Update `README.md` if it affects user-facing features
4. Add examples in `docs/DFM_PIPELINE_GUIDE.md`

### Documentation Style

- Use clear, concise language
- Include code examples
- Add "why" context, not just "how"
- Keep it up-to-date with code changes

## 🔄 Contribution Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, documented code
- Follow the style guidelines
- Add tests for new features
- Update documentation

### 3. Test Your Changes

```bash
# Run syntax checks
python -m py_compile core/*.py

# Run tests
python tests/test_dfm_pipeline.py

# Test manually if needed
python -m core.dfm_pipeline data/sample_dfm.txt
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Brief description of changes"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Add detailed description if needed
- Reference issues: "Fix #123" or "Closes #456"

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what and why
- Reference to related issues
- Screenshots for UI changes

## 🐛 Bug Reports

### Before Submitting

1. Check if the bug has already been reported
2. Try the latest version from main branch
3. Review `docs/TROUBLESHOOTING.md`

### Good Bug Report Includes

- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Minimal code example if applicable

### Template

```markdown
**Environment:**
- Python version: 3.X.X
- OS: Ubuntu 22.04 / macOS 14 / Windows 11
- Package versions: (from `pip list`)

**Description:**
Brief description of the issue

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Error Message:**
```
Paste error traceback here
```

**Additional Context:**
Any other relevant information
```

## ✨ Feature Requests

We welcome feature requests! Please:

1. Check existing issues to avoid duplicates
2. Describe the problem you're trying to solve
3. Propose a solution if you have one
4. Explain the use case and benefits
5. Consider implementing it yourself (we'll help!)

## 🎯 Priority Areas

We're particularly interested in contributions for:

1. **Performance Optimization**
   - Faster embedding generation
   - Efficient batch processing
   - Memory usage improvements

2. **Enhanced Extraction**
   - Better OCR support
   - Table extraction improvements
   - Multi-language support

3. **Model Integration**
   - Support for more LLM backends
   - Custom model fine-tuning guides
   - Smaller, faster models

4. **Documentation**
   - More examples and tutorials
   - Video guides
   - Translation to other languages

5. **Testing**
   - More comprehensive tests
   - Integration tests with real documents
   - Performance benchmarks

## 📋 Code Review Process

After submitting a PR:

1. **Automated checks** will run (syntax, tests)
2. **Maintainer review** within 1-2 weeks
3. **Feedback** if changes are needed
4. **Approval and merge** when ready

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Backward compatibility

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and discussions
- **Discussions**: General questions and ideas

## 📜 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Recognition

All contributors will be:
- Listed in `CONTRIBUTORS.md` (if you make one)
- Mentioned in release notes for significant contributions
- Given credit in the documentation

## ❓ Questions?

If you have questions:
1. Check `docs/TROUBLESHOOTING.md`
2. Search existing GitHub issues
3. Create a new issue with the "question" label
4. Be specific and provide context

---

Thank you for contributing to making DFM rule extraction better! 🚀
