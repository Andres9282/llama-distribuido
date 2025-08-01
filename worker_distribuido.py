# worker_distribuido.py

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_id = "meta-llama/Llama-2-7b-hf"
    hf_token = os.getenv("HF_TOKEN", "hf_...")  # reemplaza por tu token o usa variable de entorno

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

    # Rank 0 recibirá el prompt; los demás esperarán broadcast
    input_ids = None
    if rank == 0:
        prompt = "Explica brevemente la teoría de la relatividad."  # temporal, luego viene desde orquestador
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = dist.broadcast_object_list([input_ids], src=0)[0]

    with torch.no_grad():
        output = model.generate(input_ids.to(model.device), max_new_tokens=100)

    if rank == 0:
        print("\n🧠 Respuesta generada:\n")
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
