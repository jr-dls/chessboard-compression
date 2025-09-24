import os
import io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess.pgn
import zstandard as zstd
from multiprocessing import Process, Queue, Event, set_start_method

# =====================
# Configuración global
# =====================
BATCH_SIZE = 64
SAVE_EVERY = 10_000      # guardar cada 10k batches
BUFFER_SIZE = 10_000     # ejemplos en buffer de shuffle
NUM_WORKERS = 8          # procesos lectores en paralelo
NUM_THREADS = 4          # threads internos de Torch
OUTPUT_DIR = "C:/Users/IN_CAP02/Documents/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Modelo
# =====================
class CompletionNet(nn.Module):
    def __init__(self, in_channels=2, num_pieces=12, num_squares=64):
        super().__init__()
        self.input_dim = in_channels * num_pieces * num_squares
        hidden = self.input_dim * 2
        output_dim = num_pieces * num_squares

        self.fc1 = nn.Linear(self.input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 2*12*64)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits

# =====================
# Funciones del usuario
# (a implementar por ti)
# =====================
def generate_input(board) -> np.ndarray:
    """
    Genera un input 2x12x64 para la red neuronal a partir de un objeto chess.Board
    Canal 0: información negativa (pieza no puede estar en la casilla)
    Canal 1: información positiva (pieza sí está en la casilla)
    """
    input_arr = np.zeros((2, 12, 64), dtype=np.float32)

    # Listas de tuplas (pieza_idx, casilla)
    negative_info = []
    positive_info = []

    # Recorremos las 64 casillas
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            # Casilla vacía → todas las piezas son negativas
            for piece_idx in range(12):
                negative_info.append((piece_idx, square))
        else:
            # Casilla ocupada
            # Obtenemos el índice de la pieza (0..11)
            # Asumimos: piezas ordenadas [P,W,N,B,R,Q,...] o como quieras definir
            # chess.Piece.color: True=blanco, False=negro
            # chess.Piece.piece_type: 1..6 (K,Q,R,B,N,P)
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = color_offset + (piece.piece_type - 1)

            positive_info.append((piece_idx, square))

            # Todas las demás piezas no pueden estar en esta casilla
            for other_piece_idx in range(12):
                if other_piece_idx != piece_idx:
                    negative_info.append((other_piece_idx, square))

    # Selección aleatoria de un subconjunto de cada lista
    num_neg = random.randint(0, len(negative_info)) if negative_info else 0
    num_pos = random.randint(0, len(positive_info)) if positive_info else 0

    neg_subset = random.sample(negative_info, num_neg) if num_neg > 0 else []
    pos_subset = random.sample(positive_info, num_pos) if num_pos > 0 else []

    # Marcar en el arreglo
    for piece_idx, square in neg_subset:
        input_arr[0, piece_idx, square] = 1.0
    for piece_idx, square in pos_subset:
        input_arr[1, piece_idx, square] = 1.0

    return input_arr

def generate_output(board) -> np.ndarray:
    """
    Genera el output 12x64 para entrenamiento.
    1 si la pieza está en esa casilla, 0 si no.
    """
    output_arr = np.zeros((12, 64), dtype=np.float32)

    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = color_offset + (piece.piece_type - 1)
            output_arr[piece_idx, square] = 1.0

    return output_arr

# =====================
# Lector de partidas
# =====================
def game_reader_worker(path, queue: Queue, stop_event: Event, sample_prob=0.01):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        while not stop_event.is_set():
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            board = game.board()

            # Recorrer jugadas principales
            for move in game.mainline_moves():
                # procesar posición actual antes del movimiento
                if random.random() < sample_prob:
                    try:
                        X = generate_input(board)
                        Y = generate_output(board)
                        queue.put((X, Y))
                    except Exception as e:
                        print("[reader] error generando ejemplo:", e)

                board.push(move)

            # procesar la posición final (después de la última jugada)
            if random.random() < sample_prob:
                try:
                    X = generate_input(board)
                    Y = generate_output(board)
                    queue.put((X, Y))
                except Exception as e:
                    print("[reader] error generando ejemplo:", e)

    print("[reader] terminado:", path)

# =====================
# Trainer loop con buffer-shuffle
# =====================
def trainer_loop(queue: Queue, stop_event: Event, device="cpu"):
    torch.set_num_threads(NUM_THREADS)
    device = torch.device(device)
    # Para entrenar desde cero:
    # model = CompletionNet().to(device)
    # Para entrenar desde checkpoint:
    model = CompletionNet()
    ckpt = torch.load("C:/Users/IN_CAP02/Documents/ResultadosNN/PrimerEntrenamiento/ckpt_00050000.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    buffer_inputs, buffer_targets = [], []
    batch_count, total_seen = 0, 0

    ema_loss = None

    while True:
        try:
            item = queue.get(timeout=5.0)
        except:
            if stop_event.is_set() and queue.empty():
                break
            else:
                continue

        # --- check sentinel ---
        if item is None:
            # indica que no habrá más datos
            break

        in_np, out_np = item
        buffer_inputs.append(in_np.astype(np.float32))
        buffer_targets.append(out_np.astype(np.float32))
        total_seen += 1

        # limitar tamaño del buffer
        if len(buffer_inputs) > BUFFER_SIZE:
            buffer_inputs = buffer_inputs[-BUFFER_SIZE:]
            buffer_targets = buffer_targets[-BUFFER_SIZE:]

        # entrenar cuando haya suficientes ejemplos
        if len(buffer_inputs) >= BATCH_SIZE:
            idxs = np.random.choice(len(buffer_inputs), size=BATCH_SIZE, replace=False)
            X = np.stack([buffer_inputs[i] for i in idxs], axis=0)
            Y = np.stack([buffer_targets[i] for i in idxs], axis=0)
            # eliminar de buffer
            for i in sorted(idxs, reverse=True):
                buffer_inputs.pop(i)
                buffer_targets.pop(i)

            X_t = torch.from_numpy(X).to(device)
            Y_t = torch.from_numpy(Y).to(device).view(X.shape[0], -1)

            model.train()
            optimizer.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, Y_t)
            loss.backward()
            optimizer.step()

            if ema_loss == None:
                ema_loss = loss.item()

            ema_loss = 0.99 * ema_loss + 0.01 * loss.item()

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"[trainer] batch {batch_count}, ema loss={ema_loss:.6f}, loss={loss.item():.6f}, seen={total_seen}")
            if batch_count % SAVE_EVERY == 0:
                path = os.path.join(OUTPUT_DIR, f"ckpt_{batch_count:08d}.pth")
                torch.save(model.state_dict(), path)
                print(f"[trainer] checkpoint guardado: {path}")

    print("[trainer] terminado, batches totales:", batch_count)


# =====================
# Main
# =====================
def main():
    set_start_method("spawn", force=True)
    data_path = "lichess_db_standard_rated_2023-03.pgn.zst"  # <-- cámbialo a tu archivo

    queue = Queue(maxsize=5000)
    stop_event = Event()

    # procesos lectores
    readers = [
        Process(target=game_reader_worker, args=(data_path, queue, stop_event))
        for _ in range(NUM_WORKERS)
    ]
    for p in readers:
        p.start()

    # proceso trainer
    trainer = Process(target=trainer_loop, args=(queue, stop_event, "cpu"))
    trainer.start()

    try:
        trainer.join()
    except KeyboardInterrupt:
        print("Parando entrenamiento…")
    finally:
        stop_event.set()
        for p in readers:
            p.join()

        # enviar sentinel para indicar fin de datos
        for _ in range(NUM_WORKERS):
            queue.put(None)

        trainer.join()


if __name__ == "__main__":
    main()
