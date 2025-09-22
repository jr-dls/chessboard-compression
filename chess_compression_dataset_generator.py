import chess_compression
from chess_compression import encode_chessboard

import chess.pgn
import zstandard as zstd
import io
import random
import numpy as np
import datetime
import os

# --- Configuración ---
filename = "lichess_db_standard_rated_2023-03.pgn.zst"
output_dir = "E:/dataset"
os.makedirs(output_dir, exist_ok=True)

CHUNK_SIZE = 50_000       # Número de tuplas por archivo
SAVE_PROB = 0.1          # 1/10 probabilidad de guardar cada tupla
STOP_TIME = datetime.datetime.now().replace(hour=5, minute=0, second=0, microsecond=0)
if STOP_TIME < datetime.datetime.now():  # si ya pasó, usar mañana
    STOP_TIME += datetime.timedelta(days=1)

# --- Variables de control ---
numero_piezas_data = []
existencia_pieza_data = []

chunk_id = 0
total_saved = 0
reached_stop = False  # bandera para evitar doble guardado

def save_chunk(numero_piezas_data, existencia_pieza_data, chunk_id, output_dir):
    """Guardar un chunk comprimido en disco, separado por tipo de ejemplo"""
    if not numero_piezas_data and not existencia_pieza_data:
        return

    # --- numero_piezas ---
    if numero_piezas_data:
        inputs_num, piece_indices_num, outputs_num = zip(*numero_piezas_data)
        inputs_num = np.stack(inputs_num, axis=0).astype(np.float32)
        piece_indices_num = np.array(piece_indices_num, dtype=np.int64)
        outputs_num = np.stack(outputs_num, axis=0).astype(np.float32)
    else:
        inputs_num = np.zeros((0, 1560), dtype=np.float32)
        piece_indices_num = np.zeros((0,), dtype=np.int64)
        outputs_num = np.zeros((0, 11), dtype=np.float32)

    # --- existencia_pieza ---
    if existencia_pieza_data:
        inputs_ex, piece_indices_ex, squares_ex, labels_ex = zip(*existencia_pieza_data)
        inputs_ex = np.stack(inputs_ex, axis=0).astype(np.float32)
        piece_indices_ex = np.array(piece_indices_ex, dtype=np.int64)
        squares_ex = np.array(squares_ex, dtype=np.int64)
        labels_ex = np.array(labels_ex, dtype=np.float32)
    else:
        inputs_ex = np.zeros((0, 1560), dtype=np.float32)
        piece_indices_ex = np.zeros((0,), dtype=np.int64)
        squares_ex = np.zeros((0,), dtype=np.int64)
        labels_ex = np.zeros((0,), dtype=np.float32)

    # --- Guardar ---
    path = os.path.join(output_dir, f"chunk_{chunk_id:05d}.npz")
    np.savez_compressed(
        path,
        inputs_num=inputs_num,
        piece_indices_num=piece_indices_num,
        outputs_num=outputs_num,
        inputs_ex=inputs_ex,
        piece_indices_ex=piece_indices_ex,
        squares_ex=squares_ex,
        labels_ex=labels_ex,
    )
    print(f"[+] Guardado {inputs_num.shape[0]}+{inputs_ex.shape[0]} ejemplos en {path}")

# --- Procesamiento ---
with open(filename, "rb") as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        game_iter = iter(lambda: chess.pgn.read_game(text_stream), None)

        for game_id, game in enumerate(game_iter, start=1):
            # --- Verificar hora ---
            if datetime.datetime.now() >= STOP_TIME:
                print(f"[!] Hora límite alcanzada ({STOP_TIME}). Finalizando...")
                total_saved += len(numero_piezas_data) + len(existencia_pieza_data)
                save_chunk(numero_piezas_data, existencia_pieza_data, chunk_id, output_dir)
                reached_stop = True
                break

            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue  # partida vacía

            # --- Seleccionar una posición al azar ---
            pos_index = random.randint(0, len(moves))  # 0 = posición inicial
            for move in moves[:pos_index]:
                board.push(move)

            # --- Llamar encoder para esa posición ---
            encoding_data = []
            encode_chessboard(board, enc=None, collect_data=True, encoding_data=encoding_data)

            # --- Filtrar tuplas con probabilidad 1/100 ---
            for tup in encoding_data:
                if random.random() < SAVE_PROB:
                    if tup[1] == "numero_piezas":
                        # (input, "numero_piezas", piece_index, output)
                        numero_piezas_data.append((tup[0], tup[2], tup[3]))
                    elif tup[1] == "existencia_pieza":
                        # (input, "existencia_pieza", piece_index, square, label)
                        existencia_pieza_data.append((tup[0], tup[2], tup[3], tup[4]))

                    if len(numero_piezas_data) + len(existencia_pieza_data) >= CHUNK_SIZE:
                        total_saved += len(numero_piezas_data) + len(existencia_pieza_data)
                        save_chunk(numero_piezas_data, existencia_pieza_data, chunk_id, output_dir)
                        numero_piezas_data.clear()
                        existencia_pieza_data.clear()
                        chunk_id += 1

            if game_id % 1000 == 0:
                print(f"Procesadas {game_id} partidas, {total_saved} tuplas guardadas...")

# --- Guardar lo que falte (si no fue por STOP_TIME) ---
if not reached_stop:
    total_saved += len(numero_piezas_data) + len(existencia_pieza_data)
    save_chunk(numero_piezas_data, existencia_pieza_data, chunk_id, output_dir)

print(f"Finalizado. Total de tuplas guardadas: {total_saved}")
