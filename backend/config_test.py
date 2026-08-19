from utils.config import settings

print("=" * 40)
print("TruthLens AI Configuration")
print("=" * 40)

print(f"Model        : {settings.GROQ_MODEL}")
print(f"Backend Host : {settings.BACKEND_HOST}")
print(f"Backend Port : {settings.BACKEND_PORT}")
print(f"Top K        : {settings.TOP_K}")
print(f"Max Loops    : {settings.MAX_AGENT_ITERATIONS}")

if settings.GROQ_API_KEY:
    print("Groq API     : Loaded")
else:
    print("Groq API     : Not Set")