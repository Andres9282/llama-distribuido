# orquestador.py

import subprocess

# ✅ Prompt fijo por ahora (más adelante vendrá del cliente)
prompt = "Explica brevemente qué es la inteligencia artificial."

print("🧠 ORQUESTADOR | Preparando ejecución distribuida...")
print(f"➡️  Prompt: {prompt}\n")

# 🚀 Lanza el modelo distribuido (solo en rank 0; los demás deben ejecutarse manualmente o vía SSH)
# Este archivo debe estar ya sincronizado en todas las máquinas

try:
    subprocess.run(["accelerate", "launch", "worker_distribuido.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Error al ejecutar modelo distribuido: {e}")
