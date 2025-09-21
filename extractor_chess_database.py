# Modificación para bases de datos .bz2
# Librerías necesarias para descomprimir.
import bz2

# # Modificación para bases de datos .zst
# # Librerías necesarias para descomprimir.
# import zstandard as zstd
# import io

# Librerías necesarias para descomprimir.
import chess.pgn
import numpy as np

# --- Configuración ---
filename = "lichess_db_standard_rated_2013-01.pgn.bz2"

# Arrays por casilla para cada tipo de pieza
piece_counters = {
    chess.Piece(chess.KING, chess.WHITE): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.KING, chess.BLACK): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.QUEEN, chess.WHITE): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.QUEEN, chess.BLACK): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.ROOK, chess.WHITE): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.ROOK, chess.BLACK): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.BISHOP, chess.WHITE): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.BISHOP, chess.BLACK): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.KNIGHT, chess.WHITE): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.KNIGHT, chess.BLACK): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.PAWN, chess.WHITE): np.zeros(64, dtype=np.int64),
    chess.Piece(chess.PAWN, chess.BLACK): np.zeros(64, dtype=np.int64),
}

# Máximos teóricos por tipo (según reglas de promoción)
max_pieces = {
    chess.Piece(chess.KING, chess.WHITE): 1,
    chess.Piece(chess.KING, chess.BLACK): 1,
    chess.Piece(chess.QUEEN, chess.WHITE): 9,
    chess.Piece(chess.QUEEN, chess.BLACK): 9,
    chess.Piece(chess.ROOK, chess.WHITE): 10,
    chess.Piece(chess.ROOK, chess.BLACK): 10,
    chess.Piece(chess.BISHOP, chess.WHITE): 10,
    chess.Piece(chess.BISHOP, chess.BLACK): 10,
    chess.Piece(chess.KNIGHT, chess.WHITE): 10,
    chess.Piece(chess.KNIGHT, chess.BLACK): 10,
    chess.Piece(chess.PAWN, chess.WHITE): 8,
    chess.Piece(chess.PAWN, chess.BLACK): 8,
}

# Distribución de número de piezas (ejemplo: cuántas posiciones tenían 2 damas blancas, etc.)
distribution_counters = {
    piece: np.zeros(max_count + 1, dtype=np.int64)
    for piece, max_count in max_pieces.items()
    # piece: np.zeros(11, dtype=np.int64) for piece in piece_counters
}

# Contador total de posiciones analizadas
total_positions = 0

# Modificación para bases de datos .bz2 (NOTAR TABULACIÓN)
# --- Procesamiento ---
with bz2.open(filename, mode="rt", encoding="utf-8", errors="replace") as f:
    game_iter = iter(lambda: chess.pgn.read_game(f), None)

# # Modificación para bases de datos .zst (NOTAR TABULACIÓN)
# # --- Procesamiento ---
# with open(filename, "rb") as fh:
#     dctx = zstd.ZstdDecompressor()
#     # stream_reader devuelve un flujo descomprimido on the fly
#     with dctx.stream_reader(fh) as reader:
#         # lo envolvemos en un TextIOWrapper para que se comporte como texto
#         text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

#         game_iter = iter(lambda: chess.pgn.read_game(text_stream), None)

    for i, game in enumerate(game_iter, start=1):
        board = game.board()

        for move in [None] + list(game.mainline_moves()):
            if move is not None:
                board.push(move)

            total_positions += 1  # nueva posición procesada

            # Contar cuántas piezas de cada tipo hay en la posición
            piece_counts = {p: 0 for p in piece_counters}

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece in piece_counters:
                    # Incrementar mapa de piezas en casillas
                    piece_counters[piece][square] += 1
                    # Contar cuántas piezas de ese tipo/color hay en la posición
                    piece_counts[piece] += 1

            # Actualizar las distribuciones
            for piece, count in piece_counts.items():
                distribution_counters[piece][count] += 1

        if i % 10000 == 0:
            print(f"Procesadas {i} partidas...")

import csv

# --- Guardar piece_counters (frecuencia de cada pieza en cada casilla) ---
with open("piece_counters.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["piece", "square_index", "square_name", "count"])
    for piece, arr in piece_counters.items():
        for square, count in enumerate(arr):
            writer.writerow([
                piece.symbol(),         # Ejemplo: "K" (rey blanco), "q" (dama negra)
                square,                 # índice de 0 a 63
                chess.square_name(square),  # nombre tipo "a1", "h8"
                count
            ])

# --- Guardar distribution_counters (distribución de número de piezas por posición) ---
with open("distribution_counters.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["piece", "num_pieces", "positions"])
    for piece, arr in distribution_counters.items():
        for num_pieces, positions in enumerate(arr):
            writer.writerow([
                piece.symbol(),     # Ejemplo: "Q", "n", "p"
                num_pieces,         # cuántas piezas había (0..máximo posible)
                positions           # cuántas posiciones tenían esa cantidad
            ])

# --- Guardar total de posiciones ---
with open("summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["total_positions"])
    writer.writerow([total_positions])