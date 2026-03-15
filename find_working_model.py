import google.generativeai as genai
import os

genai.configure(api_key='AIzaSyBsvNnVDobbO3gH0h742VaFkPjRTuRJ5rg')

print("Listing and testing models...")
available_models = genai.list_models()
for m in available_models:
    if 'generateContent' in m.supported_generation_methods:
        print(f"Testing {m.name}...", end=" ")
        try:
            model = genai.GenerativeModel(m.name)
            response = model.generate_content('hi', generation_config={'max_output_tokens': 5})
            print(f"✅ SUCCESS")
            print(f"FOUND WORKING MODEL: {m.name}")
            # Write the result to a file so I can read it
            with open("working_model.txt", "w") as f:
                f.write(m.name)
            break
        except Exception as e:
            err = str(e)
            if "429" in err:
                print("❌ FAILED (Quota Exceeded)")
            elif "404" in err:
                print("❌ FAILED (Not Found)")
            else:
                print(f"❌ FAILED ({err[:50]}...)")
