import os
from dotenv import load_dotenv
load_dotenv()

from utils.llm_utils import get_llm_instance
import warnings
warnings.filterwarnings('ignore')

providers = ["gemini", "groq", "openai"]
for p in providers:
    try:
        llm = get_llm_instance(p)
        if llm:
            res = llm.invoke("Say hi")
            print(f"✅ {p}: Success ({res.content})")
        else:
            print(f"❌ {p}: Key missing")
    except Exception as e:
        print(f"❌ {p}: Error - {str(e)}")
