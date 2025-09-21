import arithmeticcoding
import io
import numpy as np
import chess
import csv

# --- Recuperar piece_counters ---
piece_counters = {}
with open("piece_counters.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        symbol = row["piece"]
        square_index = int(row["square_index"])
        count = int(row["count"])

        # Convertir símbolo PGN a objeto Piece
        piece = chess.Piece.from_symbol(symbol)

        if piece not in piece_counters:
            piece_counters[piece] = np.zeros(64, dtype=np.int64)

        piece_counters[piece][square_index] = count

# --- Recuperar distribution_counters ---
distribution_counters = {}
with open("distribution_counters.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        symbol = row["piece"]
        num_pieces = int(row["num_pieces"])
        positions = int(row["positions"])

        piece = chess.Piece.from_symbol(symbol)

        if piece not in distribution_counters:
            # el tamaño se ajusta dinámicamente al máximo encontrado
            distribution_counters[piece] = []

        # aseguramos que la lista tenga espacio
        arr = distribution_counters[piece]
        while len(arr) <= num_pieces:
            arr.append(0)

        arr[num_pieces] = positions

# Convertimos listas en arrays de numpy
for piece in distribution_counters:
    distribution_counters[piece] = np.array(distribution_counters[piece], dtype=np.int64)

# --- Recuperar total_positions ---
with open("summary.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    total_positions = int(next(reader)["total_positions"])


# Evita que BytesIO se "cierre" cuando BitOutputStream llame a close()
class NonClosingBytesIO(io.BytesIO):
    def close(self):
        # sólo vacía, no marca el objeto como cerrado, así getvalue() sigue funcionando
        try:
            super().flush()
        except Exception:
            pass
    def really_close(self):
        # para cerrar de verdad si lo necesitas
        super().close()


def encode_chessboard(board, enc, collect_data = False, encoding_data = None):
    """
    The input to the neural predictor contains:
    -> A board representation (a list of length 64 for each piece) of the already compressed or decompressed pieces.
    -> An equal length input that stores if the state in the board representation is really known.
    -> A list with the info about the number of pieces already compressed or decompressed.
    -> An equal lenght input that stores if the number of pieces is really known.
    A num_piece equal to zero implicates that the current_piece number is still unknown.
    The output to the neural predictor contains:
    -> A board representation with the probability of finding a piece in a position.
    -> A list for each piece with the probability distribution for the number of pieces.
    """
    # Storage of assigned squares.
    is_square_assigned = [False]*64
    if collect_data:
        neural_predictor_input = np.zeros(1560, dtype=np.float32)
    # Codificación.
    for piece_index, piece in enumerate(piece_counters.keys()):  # o la lista de piezas que quieras codificar
        # Codifica la información de una pieza específica (ej. dama blanca).

        # 1. Número de piezas de este tipo en la posición actual
        num_pieces = len(board.pieces(piece.piece_type, piece.color))

        # 2. Distribución de frecuencias (cuántas posiciones tenían 0, 1, 2... piezas)
        num_pieces_freqs = distribution_counters[piece]
        enc.write(arithmeticcoding.SimpleFrequencyTable(num_pieces_freqs), num_pieces)

        # 3. Orden de casillas por frecuencia de aparición (más frecuente primero)
        square_order = np.argsort(-piece_counters[piece])

        # 4. Codificar casilla por casilla
        encoded_count = 0
        i = 0
        while encoded_count < num_pieces:
            square = square_order[i]
            if not is_square_assigned[square]:
                # Frecuencias de aparición y no aparición en esta casilla
                true_count = piece_counters[piece][square]
                false_count = total_positions - true_count
                arreglo_falso_verdadero = [false_count, true_count]

                freqs = arithmeticcoding.SimpleFrequencyTable(arreglo_falso_verdadero)

                if board.piece_at(square) == piece:
                    enc.write(freqs, 1)
                    is_square_assigned[square] = True
                    encoded_count += 1
                else:
                    enc.write(freqs, 0)

            i += 1


def decode_chessboard(dec):
    # Chess board decompression result.
    board_result = chess.Board(fen=None)
    # Storage of assigned squares.
    is_square_assigned = [False]*64
    for piece in piece_counters.keys():  # todas las piezas que quieras decodificar
        # Decodifica la información de una pieza específica (ej. dama blanca)
        # y coloca las piezas en board_result.

        # 1. Leer número de piezas de este tipo en la posición
        num_pieces_freqs = distribution_counters[piece]
        number_of_pieces = dec.read(arithmeticcoding.SimpleFrequencyTable(num_pieces_freqs))

        # 2. Orden de casillas por frecuencia de aparición (más frecuente primero)
        square_order = np.argsort(-piece_counters[piece])

        # 3. Decodificar casilla por casilla
        encoded_count = 0
        i = 0
        while encoded_count < number_of_pieces:
            square = square_order[i]

            if not is_square_assigned[square]:
                # Frecuencias de aparición y no aparición
                true_count = piece_counters[piece][square]
                false_count = total_positions - true_count
                arreglo_falso_verdadero = [false_count, true_count]

                freqs = arithmeticcoding.SimpleFrequencyTable(arreglo_falso_verdadero)

                if dec.read(freqs) == 1:
                    board_result.set_piece_at(square, piece)
                    is_square_assigned[square] = True
                    encoded_count += 1

            i += 1
    return board_result


def test_encode_and_decode():
    # Compression of a group of FENs sequentially.
    # FENs form Bratko-Kopec Test:
    BK_Test_FENs_list_input = ["1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5",
    "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5",
    "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R",
    "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R",
    "r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1",
    "2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8",
    "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1",
    "4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8",
    "2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R",
    "3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1",
    "2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1",
    "r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1",
    "r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1",
    "rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1",
    "2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1",
    "r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1",
    "r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1",
    "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1",
    "3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1",
    "r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3",
    "3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1",
    "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1",
    "r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1",
    "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1"]

    # Bit stream and encoder initialization.
    buffer = NonClosingBytesIO()
    bitout = arithmeticcoding.BitOutputStream(buffer)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    # Execution of compression.
    for position in BK_Test_FENs_list_input:
        encode_chessboard(chess.Board(position), enc)
    # Finishing compression
    # MUY IMPORTANTE cerrar el bitout para escribir los últimos bits relevantes:
    enc.finish()
    bitout.close()
    encoded = buffer.getvalue()
    buffer.really_close()

    # board = chess.Board("rnbq1rk1/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPPQ1PPP/R3K2R")
    # print(board)
    # encoded = chess_encoding(board)

    # Printing encoded bytes.
    # print("Bits codificados:", ''.join(f"{b:08b}" for b in encoded))
    # print("Bytes codificados:", encoded)

    print("------ Prueba de compresión y decompresión ------")
    print("Número de bytes: ", len(encoded))
    print("Número de bytes promedio por posición: ", len(encoded)/len(BK_Test_FENs_list_input))

    FENs_salida = []

    # Creation of input buffer.
    buffer_in = io.BytesIO(encoded)
    bitin = arithmeticcoding.BitInputStream(buffer_in)
    # Decoder initialization.
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    for _ in range(len(BK_Test_FENs_list_input)):
        FENs_salida.append(decode_chessboard(dec).board_fen())
    # Clossure of input.
    bitin.close()

    are_equal = True
    for i in range(len(BK_Test_FENs_list_input)):
        if BK_Test_FENs_list_input[i] != FENs_salida[i]:
            are_equal = False
    
    if are_equal:
        print("Los tableros de entrada y salida coinciden.")
    else:
        print("------ ERROR: Los tableros de entrada y salida NO coinciden. ------")
    
    print("------ Prueba de compresión y decompresión ------")

def entropy_calculation_of_chessboard(board):
    """
    The imput to this function is a chess.Board() object that stores a specific position.
    chess.Board() object can be instantiated from a FEN or extracted from a .pgn database.
    The board would be compressed 80 times and the resultant number of bytes be used
    to calculate the chessboard entropy to a 0.1 bits uncertainty.
    """

    # Bit stream and encoder initialization.
    buffer = NonClosingBytesIO()
    bitout = arithmeticcoding.BitOutputStream(buffer)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    # Execution of compression.
    for _ in range(80):
        encode_chessboard(board, enc)
    # Finishing compression
    # MUY IMPORTANTE cerrar el bitout para escribir los últimos bits relevantes:
    enc.finish()
    bitout.close()
    encoded = buffer.getvalue()
    buffer.really_close()

    return len(encoded)/10

test_encode_and_decode()
board = chess.Board("r1b2r1k/4qp1p/p1Nppb1Q/4nP2/1p2P3/2N5/PPP4P/2KR1BR1")
print("")
print(f"Número de bits posición: {entropy_calculation_of_chessboard(board):.1f}")
print(board)