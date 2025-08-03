# 🧠 LLaMA Distribuido

Sistema de inferencia distribuida para el modelo LLaMA 2 usando `torch.distributed` y `accelerate`, coordinado por una orquestadora y ejecutado por múltiples nodos que comparten el modelo de forma real.

---

## ⚙️ Requisitos por nodo

- Python 3.9–3.11
- Git
- Tailscale (para red privada entre máquinas)
- GPU con soporte CUDA
- Token de Hugging Face con acceso a `meta-llama/Llama-2-7b-hf`
- Entorno virtual (`venv`)
- Configuración individual de Accelerate

---

## 🧭 Configuración por nodo

```bash
# Clonar el repositorio
git clone https://github.com/Andres9282/llama-distribuido.git
cd llama-distribuido

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install torch transformers accelerate
