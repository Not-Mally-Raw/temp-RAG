import ast

try:
    with open('image_pipeline_v4_multimodal.py', 'r') as f:
        content = f.read()
    # Remove the pip install line that would cause issues
    content = content.replace('!pip install timm sentence-transformers', '# pip install timm sentence-transformers')
    ast.parse(content)
    print('✅ Syntax is valid')
    
    # Check if DualTaskTrainer class exists and has correct __init__
    if 'class DualTaskTrainer:' in content:
        print('✅ DualTaskTrainer class found')
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def __init__(self, model, tokenizer, vocab, device=' in line:
                print(f'✅ Found correct __init__ signature at line {i+1}')
                break
        else:
            print('❌ DualTaskTrainer __init__ signature not found')
    else:
        print('❌ DualTaskTrainer class not found')
        
except SyntaxError as e:
    print(f'❌ Syntax error: {e}')
except Exception as e:
    print(f'❌ Error: {e}')