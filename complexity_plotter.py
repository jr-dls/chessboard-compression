#%% -------------------------------------------------------------------------------------
#   Proyecto Análisis de la complejidad de posiciones de partidas de ajedrez
#   Programa de Estudios Superiores 2025
#   Programación I
#   Banco de Guatemala
#   -------------------------------------------------------------------------------------

# Libraries to import
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Plot function
def create_plot(csv_path : str, output_svg_path: str):
    """
    Reads complexity data from a CSV file and generates a line plot.

    Args:
        csv_path (str): The path to the input CSV file.
        output_svg_path (str): The path to save the output SVG plot.
    """
    # --- 1. Validate Input File ---
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found at '{csv_path}'")
        sys.exit(1)
    
    # --- 2. Load Data using Pandas ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: '{e}'")
        sys.exit(1)
    
    # --- 3. Determine Columns for Plotting ---
    x_col = 'half_move'
    y_col = None
    plot_title = 'Complejidad de posición vs. Turno'
    y_label = 'Complejidad o Entropía (bits)'

    if 'average_complexity' in df.columns:
        y_col = 'average_complexity'
        plot_title = 'Complejidad media de posición vs. Turno'
        print("Detected 'average_complexity' column for plotting.")
    elif 'complexity' in df.columns:
        y_col = 'complexity'
        # Try to use game index from filename for title
        try:
            base_name = os.path.basename(csv_path)
            game_index = ''.join(filter(str.isdigit, base_name))
            if game_index:
                plot_title = f'Complejidad vs. Turno para Partida {game_index}'
        except:
            pass # Conservar título por omisión si lo anterior falla
        print("Detected 'complexity' column for plotting.")

    if y_col is None or x_col not in df.columns:
        print(f"Error: CSV file must contain '{x_col}' column and either a 'complexity' or 'average_complexity' column.")
        sys.exit(1)
    
    # --- 4. Create the Plot using Seaborn ---

    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14,7))

    plot = sns.lineplot(data=df, x=x_col, y=y_col, markers='o', linestyle='-')

    plot.set_title(plot_title, fontsize=16)
    plot.set_xlabel('Número de Turno', fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)

    # Try some plot limits to improve visibility
    #plt.xlim(left=0)
    #plt.ylim(bottom=df[y_col].min() - 5 if not df[y_col].empty else 0)

    # --- 5. Save the Output ---
    try:
        # Get the directory of the output path
        output_dir = os.path.dirname(output_svg_path)

        # If the output path has a directory, and it doesn't exist, create it.
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        plt.savefig(output_svg_path, format='svg', bbox_inches='tight')
        print(f"Successfully save plot to {output_svg_path}")
    except Exception as e:
        print(f"Error saving SVG file: {e}")
        sys.exit(1)
    finally:
        # Missing.  Ensure the plot is closed to finalise operations
        plt.close(fig)
    
    #%% ---------------------------------------------------------------------------------
    #   Main program entry
    #   ---------------------------------------------------------------------------------

    if __name__ == '__main__':

        parser = argparse.ArgumentParser(
            description='Generate a line plot from a chess complexity CSV file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
        parser.add_argument("input_csv", help="Path to the input .csv file.")
        parser.add_argument("output_svg", help="Path to save the output .svg plot.")

        args = parser.parse_args()
        create_plot(args.input_csv, args.output_svg)
