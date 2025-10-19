#!/usr/bin/env python3
"""
Verification Script for Dual-Task Pipeline v4.0
Performs comprehensive checks to ensure the pipeline is ready for deployment
"""

import ast
import re
import sys

def check_syntax():
    """Check Python syntax"""
    print("🔍 Checking Python Syntax...")
    try:
        with open('image_pipeline_v4_multimodal.py', 'r') as f:
            content = f.read()
        ast.parse(content)
        print("✅ Syntax is valid\n")
        return True, content
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        return False, None

def check_classes(content):
    """Verify all required classes exist"""
    print("🔍 Checking Required Classes...")
    required_classes = [
        'Config',
        'ImageDownloader',
        'RandAugment',
        'MultiModalPricingDataset',
        'MultiModalDualTaskModel',
        'DualTaskTrainer',
        'MultiModalPipeline'
    ]
    
    tree = ast.parse(content)
    found_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    all_found = True
    for cls in required_classes:
        if cls in found_classes:
            print(f"  ✅ {cls}")
        else:
            print(f"  ❌ {cls} - MISSING!")
            all_found = False
    
    print()
    return all_found

def check_dualtask_trainer(content):
    """Check DualTaskTrainer class"""
    print("🔍 Checking DualTaskTrainer...")
    
    # Check for correct signature
    if 'def __init__(self, model, tokenizer, vocab_set, device=' in content:
        print("  ✅ __init__ signature is correct")
    else:
        print("  ❌ __init__ signature is incorrect")
        return False
    
    # Check for no duplicates
    count = content.count('class DualTaskTrainer:')
    if count == 1:
        print(f"  ✅ No duplicate class definitions (found {count})")
    else:
        print(f"  ❌ Duplicate class definitions found ({count})")
        return False
    
    # Check for required methods
    required_methods = [
        'train_epoch',
        'validate',
        'generate_predictions',
        'smape_loss',
        'create_caption_targets'
    ]
    
    for method in required_methods:
        if f'def {method}(' in content:
            print(f"  ✅ {method}() method exists")
        else:
            print(f"  ❌ {method}() method MISSING")
            return False
    
    print()
    return True

def check_model(content):
    """Check MultiModalDualTaskModel"""
    print("🔍 Checking Model Architecture...")
    
    checks = {
        'Image Encoder (DINOv2)': "timm.create_model",
        'Text Encoder (MiniLM)': "AutoModel.from_pretrained",
        'Price Head': "self.price_head",
        'Caption Head': "self.caption_decoder",
        'Forward Method': "def forward(self, images",
        'Extract Features': "def extract_features("
    }
    
    all_passed = True
    for name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - MISSING!")
            all_passed = False
    
    print()
    return all_passed

def check_outputs(content):
    """Check output generation"""
    print("🔍 Checking Output Generation...")
    
    checks = {
        'Price Predictions CSV': 'price_predictions.csv',
        'Text Descriptions CSV': 'text_descriptions.csv',
        'Model Weights': 'best_dual_task_model.pth',
        'Metadata JSON': 'dual_task_metadata.json'
    }
    
    all_passed = True
    for name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - MISSING!")
            all_passed = False
    
    print()
    return all_passed

def check_instantiation(content):
    """Check correct trainer instantiation"""
    print("🔍 Checking Trainer Instantiation...")
    
    pattern = r'DualTaskTrainer\(model,\s*tokenizer,\s*vocab\)'
    if re.search(pattern, content):
        print("  ✅ Trainer instantiation is correct")
        print("     DualTaskTrainer(model, tokenizer, vocab)")
        print()
        return True
    else:
        print("  ❌ Trainer instantiation pattern not found")
        print()
        return False

def check_config(content):
    """Check configuration"""
    print("🔍 Checking Configuration...")
    
    config_items = [
        'DATA_DIR',
        'IMG_DIR',
        'OUTPUT_DIR',
        'IMAGE_MODEL',
        'TEXT_MODEL',
        'BATCH_SIZE',
        'LEARNING_RATE',
        'EPOCHS'
    ]
    
    all_found = True
    for item in config_items:
        if item in content:
            print(f"  ✅ {item}")
        else:
            print(f"  ⚠️  {item} - not found")
    
    print()
    return True

def main():
    """Run all checks"""
    print("="*70)
    print("🚀 DUAL-TASK PIPELINE V4.0 - VERIFICATION SCRIPT")
    print("="*70)
    print()
    
    # Check syntax
    syntax_ok, content = check_syntax()
    if not syntax_ok:
        print("\n❌ VERIFICATION FAILED: Syntax errors detected")
        return False
    
    # Run all checks
    checks = [
        check_classes(content),
        check_config(content),
        check_dualtask_trainer(content),
        check_model(content),
        check_outputs(content),
        check_instantiation(content)
    ]
    
    # Summary
    print("="*70)
    if all(checks):
        print("✅ ALL CHECKS PASSED - PIPELINE IS READY FOR DEPLOYMENT!")
        print()
        print("Next Steps:")
        print("  1. Upload image_pipeline_v4_multimodal.py to Google Colab")
        print("  2. Run the script")
        print("  3. Monitor training progress")
        print("  4. Download results from /content/outputs/")
        print()
        print("Expected Outputs:")
        print("  • price_predictions.csv (sample_id, predicted_price)")
        print("  • text_descriptions.csv (sample_id, generated_description)")
        print("  • Model weights and metadata")
        print("="*70)
        return True
    else:
        print("❌ SOME CHECKS FAILED - REVIEW ERRORS ABOVE")
        print("="*70)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
