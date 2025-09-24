# neural_codec.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# For bit stream encoding and decoding:
import arithmeticcoding
import io

# -------------------------------------------------------
# Modelo compatible: forward -> logits shaped (B, 12, 64)
# -------------------------------------------------------
class CompletionNetCompatible(nn.Module):
    def __init__(self, in_channels=2, num_pieces=12, num_squares=64, hidden_mult=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_pieces = num_pieces
        self.num_squares = num_squares
        self.input_dim = in_channels * num_pieces * num_squares
        hidden_dim = hidden_mult * self.input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Para 2 capas ocultas:
        # self.fc3 = nn.Linear(hidden_dim, num_pieces * num_squares)
        # Para 3 capas ocultas:
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_pieces * num_squares)

    def forward(self, x):
        # accept (B, 2, 12, 64) or (B, input_dim)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        # Para 2 capas ocultas:
        # logits = self.fc3(h)  # (B, 12*64)
        # Para 3 capas ocultas:
        h = F.relu(self.fc3(h))
        logits = self.fc4(h)  # (B, 12*64)

        logits = logits.view(-1, self.num_pieces, self.num_squares)  # (B, 12, 64)
        return logits

# -------------------------------------------------------
# Helpers: mapping piece index <-> chess.Piece
# index convention:
# 0..5  -> white pieces (piece_type 1..6)
# 6..11 -> black pieces (piece_type 1..6)
# piece_type mapping: 1=King,2=Queen,3=Rook,4=Bishop,5=Knight,6=Pawn
# (this matches python-chess piece_type numbering)
# -------------------------------------------------------
def piece_idx_to_piece(piece_idx: int) -> chess.Piece:
    color = chess.WHITE if piece_idx < 6 else chess.BLACK
    base = piece_idx % 6
    # our mapping uses piece_type = base + 1
    piece_type = base + 1
    return chess.Piece(piece_type, color)

def piece_to_index(piece: chess.Piece) -> int:
    color_offset = 0 if piece.color == chess.WHITE else 6
    return color_offset + (piece.piece_type - 1)

# -------------------------------------------------------
# Internal: find candidate (piece_idx, square) with minimal |p-0.5|
# respecting constraints:
#   - neg[piece_idx, square] == 0 (not already known negative)
#   - no positive info for that square (i.e. pos[:, square].any() == False)
# Returns (piece_idx, square, p) or None if none available.
# -------------------------------------------------------
def _select_most_uncertain(p_mat: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray):
    # p_mat shape (12,64) float in [0,1]
    # pos_mask, neg_mask shape (12,64) booleans (True means known)
    assert p_mat.shape == (12, 64)
    # allowed positions: neg==False and pos column all False
    allowed = (~neg_mask) & (~pos_mask.any(axis=0)[None, :])  # broadcast to (12,64)
    # get indices where allowed True
    allowed_inds = np.argwhere(allowed)
    if allowed_inds.size == 0:
        return None
    # compute distance to 0.5
    dists = np.abs(p_mat[allowed] - 0.5)
    # choose minimal
    argmin = np.argmin(dists)
    piece_idx, square = allowed_inds[argmin]
    p = float(p_mat[piece_idx, square])
    return int(piece_idx), int(square), p

# -------------------------------------------------------
# Encoder function
# model: instance of CompletionNetCompatible (already loaded, eval mode)
# board: python-chess Board object (input)
# enc: arithmetic encoder object with .write(freq_table, symbol) method
# scale: integer scale to convert float probs to integer frequencies
# -------------------------------------------------------
def encode_board(model: CompletionNetCompatible, board: chess.Board, enc, scale: int = 1_000_000, device='cpu'):
    model.to(device)
    model.eval()

    # pos_mask and neg_mask booleans
    pos_mask = np.zeros((12, 64), dtype=bool)
    neg_mask = np.zeros((12, 64), dtype=bool)

    with torch.no_grad():
        while True:
            # build input tensor from masks
            inp = np.zeros((2, 12, 64), dtype=np.float32)
            inp[0, :, :] = neg_mask.astype(np.float32)
            inp[1, :, :] = pos_mask.astype(np.float32)
            x = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1,2,12,64)

            logits = model(x)  # (1,12,64)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (12,64) floats in (0,1)

            sel = _select_most_uncertain(probs, pos_mask, neg_mask)
            if sel is None:
                break
            piece_idx, square, p = sel

            # compute integer frequencies
            true_count = int(round(p * float(scale)))
            false_count = int(round((1.0 - p) * float(scale)))

            # ensure non-zero frequencies
            if true_count == 0:
                true_count = 1
            if false_count == 0:
                false_count = 1

            freqs = [false_count, true_count]  # index 0 -> 'no', index 1 -> 'yes'
            # encode the actual symbol: does board have this piece at this square?
            actual_piece = board.piece_at(square)
            actual_idx = piece_to_index(actual_piece) if actual_piece is not None else None
            symbol = 1 if (actual_idx == piece_idx) else 0

            # arithmetic coding write
            # assumes arithmeticcoding.SimpleFrequencyTable exists and enc.write(...)
            freq_table = arithmeticcoding.SimpleFrequencyTable(freqs)
            enc.write(freq_table, symbol)

            # update masks
            if symbol == 1:
                # positive known for that piece-square
                pos_mask[piece_idx, square] = True
                # negative for all other pieces on that square
                for other in range(12):
                    if other != piece_idx:
                        neg_mask[other, square] = True
            else:
                # mark negative for that piece-square
                neg_mask[piece_idx, square] = True

    # end encode_board
    return

# -------------------------------------------------------
# Decoder function
# model: as above (must be in the same state used for encoding)
# board: an empty python-chess.Board() that will be mutated in place
# dec: arithmetic decoder object with .read(freq_table) method
# scale: same scale used in encoder
# -------------------------------------------------------
def decode_board(model: CompletionNetCompatible, dec, scale: int = 1_000_000, device='cpu'):

    board = chess.Board(fen=None)

    model.to(device)
    model.eval()

    pos_mask = np.zeros((12, 64), dtype=bool)
    neg_mask = np.zeros((12, 64), dtype=bool)

    with torch.no_grad():
        while True:
            inp = np.zeros((2, 12, 64), dtype=np.float32)
            inp[0, :, :] = neg_mask.astype(np.float32)
            inp[1, :, :] = pos_mask.astype(np.float32)
            x = torch.from_numpy(inp).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (12,64)

            sel = _select_most_uncertain(probs, pos_mask, neg_mask)
            if sel is None:
                break
            piece_idx, square, p = sel

            true_count = int(round(p * float(scale)))
            false_count = int(round((1.0 - p) * float(scale)))
            if true_count == 0:
                true_count = 1
            if false_count == 0:
                false_count = 1

            freqs = [false_count, true_count]
            freq_table = arithmeticcoding.SimpleFrequencyTable(freqs)
            bit = dec.read(freq_table)  # 0 or 1

            if bit == 1:
                # place piece on board
                piece = piece_idx_to_piece(piece_idx)
                board.set_piece_at(square, piece)
                pos_mask[piece_idx, square] = True
                for other in range(12):
                    if other != piece_idx:
                        neg_mask[other, square] = True
            else:
                neg_mask[piece_idx, square] = True

    return board

# -------------------------------------------------------
# Entropy estimator (in nats)
# model: same model
# board: ground-truth board
# returns: total_entropy (float), in natural log units
# -------------------------------------------------------
def entropy_board(model: CompletionNetCompatible, board: chess.Board, device='cpu'):
    """
    Simulate the same policy but do not encode: accumulate -log prob of chosen symbol.
    """
    model.to(device)
    model.eval()

    pos_mask = np.zeros((12, 64), dtype=bool)
    neg_mask = np.zeros((12, 64), dtype=bool)
    total_entropy = 0.0

    with torch.no_grad():
        while True:
            inp = np.zeros((2, 12, 64), dtype=np.float32)
            inp[0, :, :] = neg_mask.astype(np.float32)
            inp[1, :, :] = pos_mask.astype(np.float32)
            x = torch.from_numpy(inp).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (12,64)

            sel = _select_most_uncertain(probs, pos_mask, neg_mask)
            if sel is None:
                break
            piece_idx, square, p = sel

            # ground truth symbol
            actual_piece = board.piece_at(square)
            actual_idx = piece_to_index(actual_piece) if actual_piece is not None else None
            symbol = 1 if (actual_idx == piece_idx) else 0

            # numerical safeguards
            eps = 1e-12
            p = min(max(p, eps), 1.0 - eps)

            if symbol == 1:
                total_entropy += -math.log(p)    # -ln p
                # update masks same as decoder
                pos_mask[piece_idx, square] = True
                for other in range(12):
                    if other != piece_idx:
                        neg_mask[other, square] = True
            else:
                total_entropy += -math.log(1.0 - p)
                neg_mask[piece_idx, square] = True

    return total_entropy/math.log(2.0)

# -------------------------------------------------------
# Note:
# - This module assumes `arithmeticcoding.SimpleFrequencyTable` and encoder/decoder
#   objects with `.write(freqs_table, symbol)` and `.read(freq_table)` exist.
# - Device handling: functions accept `device` param; ensure model is on same device.
# - The candidate selection policy is "most uncertain w.r.t. 0.5" and excludes already-known negatives,
#   and any square that already has a positive known label (so you don't ask further for that square).
# - If you used a different piece-index mapping in other code, adapt piece_idx<->piece helpers.
# -------------------------------------------------------

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

def test_encode_and_decode(model):
    # Compression of a group of FENs sequentially.
    # FENs form Bratko-Kopec Test:
    # BK_Test_FENs_list_input = ["1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5"]
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
        encode_board(model,chess.Board(position),enc)
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
        FENs_salida.append(decode_board(model, dec).board_fen())
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

# If you want a tiny example of usage, uncomment and adapt:
if __name__ == "__main__":
    # load model
    model = CompletionNetCompatible()
    model.load_state_dict(torch.load("C:/Users/IN_CAP02/Documents/checkpoints/ckpt_00030000.pth", map_location="cpu"))
    model.eval()
    test_encode_and_decode(model)
    board = chess.Board("r1b2r1k/4qp1p/p1Nppb1Q/4nP2/1p2P3/2N5/PPP4P/2KR1BR1")
    print("")
    print(f"Número de bits posición: {entropy_board(model, board):.1f}")
