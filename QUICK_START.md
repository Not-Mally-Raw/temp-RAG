# Quick Start Guide - RAG-RuleSync

## Setup (One-Time)

1. **Add Groq API Key**
   ```bash
   # Edit .env file
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

2. **Verify Installation**
   ```bash
   python -m pip list | Select-String "langchain|streamlit|groq"
   ```

## Running the System

### Option 1: Streamlit UI (Recommended)
```bash
cd "c:\Users\patle\Desktop\RAG\new\RAG-RuleSync"
python -m streamlit run simple_streamlit_app.py
```

Then:
1. Open browser to http://localhost:8503
2. Upload a PDF file
3. Click "Extract"
4. Find output in `output/{pdf_name}.json`

### Option 2: Python Script
```python
import asyncio
from core.enhanced_rule_engine import EnhancedConfig, EnhancedRuleEngine

config = EnhancedConfig()
engine = EnhancedRuleEngine(config)

# Extract from text
result = asyncio.run(engine.extract_rules_parallel(
    'Wall thickness must be at least 0.8mm for injection molding.',
    'my_document.pdf'
))

print(f"Rules extracted: {result['rule_count']}")
print(f"Output file: output/my_document.json")
```

## Output Format

Every PDF extraction creates `output/{filename}.json`:

```json
{
  "source_pdf": "design_guidelines.pdf",
  "rule_count": 42,
  "rules": [
    {
      "rule_text": "Minimum wall thickness is 0.8mm for injection molding",
      "applicability_constraints": {
        "material": "any",
        "process": "injection molding",
        "feature": "wall",
        "location": "any"
      },
      "dimensional_constraints": [
        "Wall thickness: >= 0.8 mm"
      ],
      "relational_constraints": [
        "None"
      ]
    }
  ],
  "processing_time": 45.2,
  "chunks_processed": 12
}
```

## Key Features

✅ **Zero Mutation**: LLM output saved exactly as generated  
✅ **Single Source**: Mega-prompt in `prompts.py` defines all extraction logic  
✅ **Direct Persistence**: JSON files created during extraction  
✅ **No Post-Processing**: No deduplication, clustering, or enhancement  
✅ **No Defaults**: No auto-filled fields or computed values  

## Common Tasks

### Check Output Files
```powershell
Get-ChildItem output/*.json | Select-Object Name, Length, LastWriteTime
```

### View JSON Output
```powershell
Get-Content output/my_document.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Extract Multiple PDFs
```python
import asyncio
from pathlib import Path
from core.enhanced_rule_engine import EnhancedConfig, EnhancedRuleEngine

config = EnhancedConfig()
engine = EnhancedRuleEngine(config)

async def process_all():
    pdf_dir = Path("pdfs/")
    for pdf in pdf_dir.glob("*.pdf"):
        text = extract_text_from_pdf(pdf)  # Use your PDF loader
        result = await engine.extract_rules_parallel(text, pdf.name)
        print(f"✅ {pdf.name}: {result['rule_count']} rules")

asyncio.run(process_all())
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Missing GROQ_API_KEY" | Add key to `.env` file |
| No output files | Check `output/` directory was created |
| Low rule count | PDF may have few explicit manufacturing rules |
| "Rate limit exceeded" | Wait 1 minute, Groq has free tier limits |

## Configuration (Optional)

Edit `core/enhanced_rule_engine.py` → `EnhancedConfig`:

```python
class EnhancedConfig(BaseSettings):
    chunk_size: int = 1500        # Tokens per chunk
    chunk_overlap: int = 150      # Overlap between chunks
    max_rules_per_document: int = 1000  # Max rules to extract
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
```

## Support

- **Logs**: Check terminal output for extraction progress
- **Errors**: Look for `ERROR` level logs with details
- **Mega-Prompt**: View extraction instructions in `core/prompts.py`

---

**System Ready**: Start extracting rules! 🚀
