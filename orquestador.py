# orquestador.py
import os
import time

PROMPT_FILE = "prompt.txt"
prompt = "Explica brevemente qué es la inteligencia artificial."

print("🧠 ORQUESTADOR | Enviando prompt al sistema distribuido...")
print(f"➡️  Prompt: {prompt}")

# Escribir el prompt al archivo que rank 0 leerá
with open(PROMPT_FILE, "w", encoding="utf-8") as f:
    f.write(prompt)

print("📤 Prompt guardado. Puedes lanzar ahora el modelo distribuido con:")
print("    accelerate launch worker_distribuido.py")
print("⏳ Esperando respuesta...")

# Esperar a que rank 0 escriba la respuesta
RESPONSE_FILE = "respuesta.txt"
while not os.path.exists(RESPONSE_FILE):
    time.sleep(1)

# Leer y mostrar la respuesta generada
with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
    respuesta = f.read()

print("\n🧠 Respuesta generada por el modelo:\n")
print(respuesta)

# Limpieza opcional
os.remove(RESPONSE_FILE)
