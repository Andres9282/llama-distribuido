import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    # Inicializar entorno distribuido
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_id = "meta-llama/Llama-2-7b-hf"
    hf_token = os.getenv("HF_TOKEN", "hf_xxx...")  # ⚠️ Reemplaza o configura como variable de entorno

    print(f"[Rank {rank}] Inicializando nodo ({world_size} nodos en total)...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # Cargar modelo distribuido entre nodos (no device_map="auto")
    print(f"[Rank {rank}] Cargando modelo distribuido...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map={"": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).eval()

    # 🔍 Imprimir y guardar las capas asignadas a este nodo
    capa_log = [f"[Rank {rank}] Capas asignadas a este nodo:"]
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.device == model.device:
            capa_log.append(f"  - {name}")

    # Imprimir en consola (puede comentar esta línea si no se quiere ver en stdout)
    for linea in capa_log:
        print(linea)

    # Guardar en archivo local
    with open(f"debug_rank{rank}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(capa_log))

    # 🔄 Recibir prompt por broadcast desde el orquestador
    prompt_container = [None]
    dist.broadcast_object_list(prompt_container, src=0)
    prompt = prompt_container[0]

    print(f"[Rank {rank}] Prompt recibido: {prompt}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # 🧠 Generar respuesta (coordinada entre los nodos)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=100)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[Rank {rank}] Respuesta generada:\n{decoded}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
