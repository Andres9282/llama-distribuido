import socket, pickle, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Verificar argumento de shard
if len(sys.argv) < 2:
    print("❌ Uso correcto: python shard.py <shard_id>")
    sys.exit(1)

try:
    shard_id = int(sys.argv[1])
except ValueError:
    print("❌ El shard_id debe ser un número entero.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔐 Reemplaza esto con tu token de Hugging Face
hf_token = "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"  # <- Asegúrate de mantenerlo privado

# Cargar modelo y tokenizer
print("[Shard] Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_response(input_text, max_new_tokens=30):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def start_shard(host, port, shard_id):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, port))
        server.listen()
        print(f"[Shard {shard_id}] Esperando conexiones en {host}:{port}...")
        while True:
            conn, addr = server.accept()
            with conn:
                print(f"[Shard {shard_id}] Conexión desde {addr}")
                data = b""
                while True:
                    part = conn.recv(4096)
                    if not part:
                        break
                    data += part
                message = pickle.loads(data)
                prompt = message["prompt"]
                print(f"[Shard {shard_id}] Generando para prompt: {prompt}")
                response = generate_response(prompt)
                conn.sendall(pickle.dumps(response))

if __name__ == "__main__":
    port = 9001 + shard_id
    start_shard("localhost", port, shard_id)