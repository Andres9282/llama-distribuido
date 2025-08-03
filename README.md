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
```

---

## 🔧 Configurar Accelerate

```bash
accelerate config
```
Compute environment:               This machine
Machine type:                      multi-GPU
How many machines:                 Cantidad de workers
Machine rank:                      De 0 hasta que cada maquina tenga su rank
Main process IP:                   IP de Tailscale de la maquina con rank 0
Port:                              29500
Are machines on same local network? no
Rendezvous backend:                static
Use DeepSpeed?                     no
Use FSDP?                          no
Use Megatron-LM?                   no
How many GPUs on this machine:     1
What GPU(s) (by id):               0
Enable NUMA efficiency:            no
Use mixed precision:               fp16

---

## 🚀 Ejecución del sistema

### 🧠 En la orquestadora:

```bash
source venv/bin/activate
python orquestador.py
```

### ⚙️ En cada nodo de procesamiento:

```bash
source venv/bin/activate
accelerate launch worker_distribuido.py
```

### 🧪 Depuración (opcional)
Cada nodo genera un archivo como:
```bash
debug_rank*.txt
```
Contienen las capas del modelo asignadas a ese nodo. Puedes analizarlas manualmente o transferirlas a la orquestadora con:
```bash
scp usuario@nodo:/ruta/debug_rank*.txt
```

### 🧼 Limpieza
Despues de cada generacion correr:
```bash
rm respuesta.txt debug_rank*.txt
```





