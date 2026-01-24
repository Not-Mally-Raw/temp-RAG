import os
import time
import re
from openai import OpenAI
from dotenv import load_dotenv  # <--- NEW IMPORT

# Load variables from .env file into the environment
load_dotenv()  # <--- NEW CALL

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "name": "Groq (Primary)"
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "gpt-oss-120b",
        "name": "Cerebras (Secondary)"
    }
}

class LLMClient:
    def __init__(self):
        self.clients = {}
        # Initialize clients for which API keys exist
        for key, config in PROVIDERS.items():
            api_key = os.getenv(config["api_key_env"])
            
            # Debug Print (Optional - remove if you want cleaner logs)
            # print(f"DEBUG: Checking {config['name']} key... Found? {bool(api_key)}")
            
            if api_key:
                self.clients[key] = OpenAI(base_url=config["base_url"], api_key=api_key)
            else:
                print(f"⚠️ Warning: {config['name']} API Key not found in .env or environment. Skipping.")

    def _extract_wait_time(self, error_message: str) -> float:
        """Parses 'Please try again in 17m18s' from Groq errors."""
        try:
            match = re.search(r"try again in\s+((?:(\d+)m)?(\d+(?:\.\d+)?)s?)", str(error_message))
            if match:
                minutes = float(match.group(2)) if match.group(2) else 0
                seconds = float(match.group(3)) if match.group(3) else 0
                return (minutes * 60) + seconds + 5  # +5s buffer
        except Exception:
            pass
        return 20.0  # Default fallback wait

    def call(self, prompt: str) -> str:
        """
        The Master Call Method: Groq -> Cerebras -> Infinite Wait
        """
        while True:
            # ---------------------------------------------------------
            # 1. Try GROQ (Primary)
            # ---------------------------------------------------------
            if "groq" in self.clients:
                try:
                    # print("    🔵 Trying Groq...")
                    response = self.clients["groq"].chat.completions.create(
                        model=PROVIDERS["groq"]["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    # Log the failure but don't crash; move to fallback
                    # print(f"    ⚠️ Groq Failed: {str(e)[:100]}...")
                    pass

            # ---------------------------------------------------------
            # 2. Try CEREBRAS (Secondary)
            # ---------------------------------------------------------
            if "cerebras" in self.clients:
                try:
                    print("    🟣 Switching to Cerebras...")
                    response = self.clients["cerebras"].chat.completions.create(
                        model=PROVIDERS["cerebras"]["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"    ⚠️ Cerebras Failed: {str(e)[:100]}...")

            # ---------------------------------------------------------
            # 3. BOTH FAILED -> Infinite Wait Loop (Targeting Groq recovery)
            # ---------------------------------------------------------
            print("    ⛔ ALL PROVIDERS EXHAUSTED. Entering Cool-down...")
            
            # Default wait time
            wait_time = 60 
            
            # Re-attempt Groq specifically to capture the "Try again in X minutes" error
            try:
                if "groq" in self.clients:
                    self.clients["groq"].chat.completions.create(
                        model=PROVIDERS["groq"]["model"],
                        messages=[{"role": "user", "content": prompt}]
                    )
            except Exception as e:
                wait_time = self._extract_wait_time(str(e))
            
            print(f"    ⏳ Sleeping for {wait_time:.1f} seconds (waiting for Groq)...")
            time.sleep(wait_time)
            print("    🔄 Waking up and restarting Provider Chain...")
            # Loop restarts from Step 1