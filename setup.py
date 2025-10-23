from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-rag-system",
    version="1.0.0",
    author="HCL Tech Manufacturing Intelligence Team",
    author_email="manufacturing-ai@hcltech.com",
    description="Enhanced RAG system for extracting manufacturing rules from random documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hcltech/enhanced-rag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "image": [
            "tesseract>=5.3.0",
            "opencv-python>=4.8.0",
            "pytesseract>=0.3.10",
        ],
        "advanced": [
            "openai>=1.0.0",
            "cohere>=4.30.0",
            "anthropic>=0.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-analytics=pages.analytics:main",
            "rag-extract=core.implicit_rule_extractor:main",
            "rag-process=core.enhanced_rag_db:main",
        ],
    },
    include_package_data=True,
    package_data={
        "data": ["*.py", "*.txt", "*.json"],
        "docs": ["*.md", "*.rst"],
    },
    keywords="rag retrieval-augmented-generation manufacturing nlp ai machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/hcltech/enhanced-rag-system/issues",
        "Source": "https://github.com/hcltech/enhanced-rag-system",
        "Documentation": "https://github.com/hcltech/enhanced-rag-system/wiki",
        "HCL Tech": "https://www.hcltech.com/",
    },
)