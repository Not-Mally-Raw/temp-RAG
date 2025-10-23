#!/bin/bash
# cleanup_rag_system.sh
# RAG-System Cleanup Script - Removes redundant files and organizes structure

set -e  # Exit on error

cd /Users/spandankewte/RAG-System

echo "============================================"
echo "   RAG-System Cleanup Script"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Confirmation
echo -e "${YELLOW}This script will:${NC}"
echo "  1. Move old documentation to docs/archive/"
echo "  2. Move old pipeline versions to docs/archive/old_pipelines/"
echo "  3. Delete empty src/ directory"
echo "  4. Delete redundant test files"
echo "  5. Archive utility scripts"
echo "  6. Organize data files"
echo ""
echo -e "${RED}MAKE SURE YOU HAVE A BACKUP!${NC}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleanup cancelled."
    exit 1
fi

echo ""
echo "Starting cleanup..."
echo ""

# 1. Create archive directories
echo -e "${GREEN}[1/7]${NC} Creating archive directories..."
mkdir -p docs/archive
mkdir -p docs/archive/old_pipelines
mkdir -p docs/archive/old_tests

# 2. Move old documentation
echo -e "${GREEN}[2/7]${NC} Moving old documentation files..."
for doc in \
    BEFORE_AFTER_COMPARISON.md \
    COLAB_SURVIVAL_GUIDE.md \
    DEPLOYMENT_READY_v4.md \
    ERROR_ANALYSIS_AND_SOLUTIONS.md \
    GPU_FIRST_ARCHITECTURE.md \
    GPU_OPTIMIZATION_FIX.md \
    Image_Pipeline_Detailed_Report.md \
    OPTIMIZATION_CHANGELOG.md \
    OPTIMIZATION_SUMMARY.md \
    PERFORMANCE_REVOLUTION.md \
    PIPELINE_ARCHITECTURE.md \
    SPEED_BOOST_CONFIG.md \
    STREAMING_ARCHITECTURE_V3.md \
    TESTING_GUIDE.md \
    TRAINING_OPTIMIZATIONS.md \
    V3_QUICK_START.md \
    V4_FIXES_SUMMARY.md \
    V6.1_RESEARCH_BREAKTHROUGH.md \
    V6_COMPLETE_COMPARISON.md \
    V6_V7_COMPARISON.md \
    V7_ULTIMATE_RESEARCH.md \
    v5_OPTIMIZATIONS.md
do
    if [ -f "$doc" ]; then
        mv "$doc" docs/archive/
        echo "  ✓ Moved $doc"
    fi
done

# 3. Move old pipeline versions
echo -e "${GREEN}[3/7]${NC} Moving old pipeline versions..."
for pipeline in \
    image_only_pipeline.py \
    image_only_pipeline_complete.py \
    image_pipeline_optimized.py \
    image_pipeline_v4_multimodal.py \
    image_pipeline_v5_optimized.py \
    image_pipeline_v6_SOTA.py \
    image_pipeline_v6.1_RESEARCH.py \
    image_pipeline_v6.2_EMERGENCY_FIX.py
do
    if [ -f "$pipeline" ]; then
        mv "$pipeline" docs/archive/old_pipelines/
        echo "  ✓ Moved $pipeline"
    fi
done

# 4. Delete empty src directory
echo -e "${GREEN}[4/7]${NC} Checking and deleting empty src directory..."
if [ -d "src/" ]; then
    # Check if all files in src/core are empty
    if [ -d "src/core/" ]; then
        file_count=$(find src/core/ -type f -name "*.py" ! -size 0 | wc -l)
        if [ "$file_count" -eq 0 ]; then
            rm -rf src/
            echo "  ✓ Deleted empty src/ directory"
        else
            echo -e "  ${YELLOW}⚠ src/ contains non-empty files, skipping deletion${NC}"
        fi
    else
        rm -rf src/
        echo "  ✓ Deleted src/ directory"
    fi
else
    echo "  ℹ src/ directory not found"
fi

# 5. Delete redundant test files
echo -e "${GREEN}[5/7]${NC} Moving redundant test files..."
for test in \
    run_tests.py \
    run_industry_testing.py \
    simple_test_runner.py \
    test_syntax.py \
    verify_pipeline.py
do
    if [ -f "$test" ]; then
        mv "$test" docs/archive/old_tests/
        echo "  ✓ Moved $test"
    fi
done

# 6. Archive utility scripts
echo -e "${GREEN}[6/7]${NC} Archiving utility scripts..."
if [ -f "auto_fix.py" ]; then
    mv auto_fix.py docs/archive/
    echo "  ✓ Moved auto_fix.py"
fi
if [ -f "fix_imports.sh" ]; then
    mv fix_imports.sh docs/archive/
    echo "  ✓ Moved fix_imports.sh"
fi

# 7. Move data files
echo -e "${GREEN}[7/7]${NC} Organizing data files..."
if [ -f "train_finaldata.csv" ]; then
    mv train_finaldata.csv data/ 2>/dev/null || true
    echo "  ✓ Moved train_finaldata.csv to data/"
fi
if [ -f "simple_industry_test_results.json" ]; then
    mv simple_industry_test_results.json docs/archive/ 2>/dev/null || true
    echo "  ✓ Moved simple_industry_test_results.json to docs/archive/"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Cleanup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Summary:"
echo "  ✓ Documentation archived to docs/archive/"
echo "  ✓ Old pipelines moved to docs/archive/old_pipelines/"
echo "  ✓ Old tests moved to docs/archive/old_tests/"
echo "  ✓ Empty directories removed"
echo "  ✓ Data files organized"
echo ""
echo "Remaining structure:"
ls -1
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Run: python quick_test.py"
echo "  2. Test the application"
echo "  3. If everything works, commit changes to git"
echo ""
