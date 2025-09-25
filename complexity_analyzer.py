import argparse
import bz2
import chess.pgn
import pandas as pd
import numpy as np
import torch
import os
import sys
import math
from collections import defaultdict

# --- Import user-provided functions and classes ---

# Check for the existence of required tool scripts
if not os.path.exists("statistical_tools.py"):
    print("Error: `statistical_tools.py` not found. Please place it in the same directory.")
    sys.exit(1)
if not os.path.exists("nn_tools.py"):
    print("Error: `nn_tools.py` not found. Please place it in the same directory.")
    sys.exit(1)

from statistical_tools import entropy_calculation_of_chessboard
from nn_tools import entropy_board, CompletionNetCompatible

def analyze_complexity(pgn_path, output_csv, use_nn=False, model_path=None, game_index=None, max_games=None):
    """
    Computes the complexity of chess games from a bz2 compressed PGN file.

    Args:
        pgn_path (str): Path to the bz2 compressed PGN file.
        output_csv (str): Path to save the output CSV file.
        use_nn (bool): If True, use the neural network method for complexity calculation.
        model_path (str): Path to the trained PyTorch model file (required if use_nn is True).
        game_index (int, optional): The 1-based index of a specific game to analyze.
        max_games (int, optional): The maximum number of games to process from the PGN.
    """
    # --- 1. Select Complexity Function and Load Model if Needed ---
    model = None
    if use_nn:
        if not model_path or not os.path.exists(model_path):
            print(f"Error: Model file not found at '{model_path}'. The --model-path is required when using --use-nn.")
            sys.exit(1)
        print(f"Loading neural network model from {model_path}...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CompletionNetCompatible()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            # Wrap the function to match the expected signature and convert nats to bits
            # 1 nat = 1/ln(2) bits ≈ 1.44 bits
            nats_to_bits_conversion = 1 / math.log(2)
            complexity_function = lambda board: entropy_board(model, board, device=device) * nats_to_bits_conversion
            print(f"Using NN-based complexity function (entropy_board) on device: {device}")
        except Exception as e:
            print(f"Failed to load the model. Error: {e}")
            sys.exit(1)
    else:
        # Check for dependencies of the statistical tool
        for f in ["summary.csv", "piece_counters.csv", "distribution_counters.csv"]:
            if not os.path.exists(f):
                print(f"Error: Data file '{f}' required by statistical_tools.py not found.")
                sys.exit(1)
        complexity_function = entropy_calculation_of_chessboard
        print("Using statistical complexity function (entropy_calculation_of_chessboard).")

    # --- 2. Process the PGN File ---
    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at '{pgn_path}'")
        sys.exit(1)

    # Use defaultdict to store lists of complexities for each half-move
    complexities_by_half_move = defaultdict(list)
    single_game_results = []
    
    games_processed = 0
    target_game_found = False

    print(f"Starting analysis of PGN file: {pgn_path}...")
    with bz2.open(pgn_path, "rt", encoding="utf-8") as pgn_file:
        current_game_idx = 0
        while True:
            # If we've processed enough games, stop.
            if max_games is not None and games_processed >= max_games:
                print(f"Reached max_games limit of {max_games}.")
                break
            
            # Read a game from the file
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # End of file
                current_game_idx += 1
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping a malformed game at index ~{current_game_idx}. Error: {e}")
                continue

            # If a specific game is requested, skip until we find it
            if game_index is not None and current_game_idx != game_index:
                continue

            # --- 3. Analyze the positions within the game ---
            board = game.board()
            half_move = 0
            
            # Analyze starting position
            complexity = complexity_function(board.copy())
            if game_index is not None:
                single_game_results.append((half_move, complexity))
            else:
                complexities_by_half_move[half_move].append(complexity)
                
            # Analyze subsequent positions
            for move in game.mainline_moves():
                board.push(move)
                half_move += 1
                complexity = complexity_function(board.copy())
                if game_index is not None:
                    single_game_results.append((half_move, complexity))
                else:
                    complexities_by_half_move[half_move].append(complexity)

            games_processed += 1
            if game_index is not None:
                target_game_found = True
                print(f"Finished analyzing game {game_index}.")
                break # Stop after processing the target game

            if games_processed % 100 == 0:
                print(f"  ... processed {games_processed} games.")

    if game_index is not None and not target_game_found:
        print(f"Warning: Game with index {game_index} not found in the PGN file.")
        return

    print(f"\nTotal games analyzed: {games_processed}.")

    # --- 4. Prepare and Save the Results ---
    if not complexities_by_half_move and not single_game_results:
        print("No data was collected. Output file will not be created.")
        return

    df = None
    if game_index is not None:
        # Create DataFrame for a single game
        df = pd.DataFrame(single_game_results, columns=['half_move', 'complexity'])
    else:
        # Calculate average complexity for all games
        avg_results = []
        for move, complexities in sorted(complexities_by_half_move.items()):
            avg_complexity = np.mean(complexities)
            avg_results.append((move, avg_complexity))
        df = pd.DataFrame(avg_results, columns=['half_move', 'average_complexity'])

    # Save to CSV
    try:
        df.to_csv(output_csv, index=False, float_format='%.3f')
        print(f"Successfully saved results to {output_csv}")
    except Exception as e:
        print(f"Error saving file to {output_csv}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the complexity of chess positions from a bz2 compressed PGN file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pgn_file", help="Path to the input .pgn.bz2 file.")
    parser.add_argument("output_csv", help="Path to the output .csv file.")
    parser.add_argument("--use-nn", action="store_true", help="Use the neural network model for entropy calculation.")
    parser.add_argument("--model-path", type=str, default="chess_model.pth", help="Path to the .pth model file (required for --use-nn).")
    parser.add_argument("--game-index", type=int, help="Analyze only the game at this specific index (1-based).")
    parser.add_argument("--max-games", type=int, help="Limit the analysis to the first N games.")
    
    args = parser.parse_args()

    analyze_complexity(
        pgn_path=args.pgn_file,
        output_csv=args.output_csv,
        use_nn=args.use_nn,
        model_path=args.model_path,
        game_index=args.game_index,
        max_games=args.max_games
    )
