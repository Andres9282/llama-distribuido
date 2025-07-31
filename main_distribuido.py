import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_id = "meta-llama/Llama-2-7b-hf"
    hf_token = os.getenv("HF_TOKEN") or "hf_..."

    def log(msg):
        if rank == 0:
            print(msg)

    log(f"[Rank {rank}] Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    log(f"[Rank {rank}] Cargando modelo distribuido...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).eval()

    # Paso 1: Rank 0 define el prompt
    prompt = "Explícame la teoría de la relatividad en pocas palabras." if rank == 0 else None

    # Paso 2: Rank 0 lo convierte en input_ids
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids if rank == 0 else None

    # Paso 3: Enviar input_ids a todos los procesos
    input_ids = dist.broadcast_object_list([input_ids], src=0)[0]

    # Paso 4: Generar respuesta de forma distribuida
    with torch.no_grad():
        output = model.generate(input_ids.to(model.device), max_new_tokens=100)

    # Paso 5: Solo rank 0 imprime la respuesta
    if rank == 0:
        print("\n🧠 Respuesta:\n")
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
