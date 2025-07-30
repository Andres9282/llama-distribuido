import socket, pickle

# Configuración de shards disponibles
shards = [
    {"host": "localhost", "port": 9001},
    {"host": "localhost", "port": 9002}
]

# Control para round-robin
current_shard_index = 0

def enviar_prompt_al_shard(prompt):
    global current_shard_index

    # Elegir shard (round-robin)
    shard = shards[current_shard_index]
    current_shard_index = (current_shard_index + 1) % len(shards)

    print(f"[Orquestador] Enviando prompt a Shard en {shard['host']}:{shard['port']}")

    # Crear socket y enviar prompt serializado
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((shard["host"], shard["port"]))
        mensaje = pickle.dumps({"prompt": prompt})
        s.sendall(mensaje)

        # Recibir respuesta
        data = b""
        while True:
            part = s.recv(4096)
            if not part:
                break
            data += part
        respuesta = pickle.loads(data)
    
    return respuesta

def main():
    print("=== 🧠 Orquestador LLaMA Distribuido ===")
    while True:
        prompt = input("📝 Ingresa tu prompt ('salir' para terminar): ")
        if prompt.lower() == "salir":
            break
        respuesta = enviar_prompt_al_shard(prompt)
        print(f"\n🧠 Respuesta:\n{respuesta}\n")

if __name__ == "__main__":
    main()