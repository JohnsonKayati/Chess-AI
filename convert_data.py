import numpy as np
import chess.pgn
import pickle
import zstandard as zstd
import re

def board_to_tensor(board):
    """
    Converts a chess board to an 8x8x14 tensor.
    """
    tensor = np.zeros((8, 8, 14), dtype=np.float32)
    p_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            #print(f"Square {chess.square_name(square)}: {piece.symbol()} ({'White' if piece.color else 'Black'})")
            channel = p_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            row, col = divmod(square, 8)
            tensor[row, col, channel] = 1

    # Set castling rights for White
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[0, 4, 12] = 1  # King square (e1)
        tensor[0, 7, 12] = 1  # Kingside rook square (h1)
    else:
        tensor[0, 4, 12] = 0
        tensor[0, 7, 12] = 0

    # Check for White queenside castling
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[0, 4, 12] = 1  # King square (e1)
        tensor[0, 0, 12] = 1  # Queenside rook square (a1)
    else:
        tensor[0, 4, 12] = 0
        tensor[0, 0, 12] = 0

    # Check for Black kingside castling
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[7, 4, 13] = 1  # King square (e8)
        tensor[7, 7, 13] = 1  # Kingside rook square (h8)
    else:
        tensor[7, 4, 13] = 0
        tensor[7, 7, 13] = 0

    # Check for Black queenside castling
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[7, 4, 13] = 1  # King square (e8)
        tensor[7, 0, 13] = 1  # Queenside rook square (a8)
    else:
        tensor[7, 4, 13] = 0
        tensor[7, 0, 13] = 0

    # Set castling rights for Black
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[7, 4, 13] = 1  # King square
        tensor[7, 7, 13] = 1  # Rook square
    else:
        tensor[7, 4, 13] = 0
        tensor[7, 7, 13] = 0

    return tensor

import chess.pgn

def extract_data_in_chunks(pgn_file_path, output_file, chunk_size=1000):
    """
    Process PGN file in chunks and save to disk using pickle to avoid memory issues.
    """
    chunk = []
    with open(pgn_file_path, 'r', encoding="utf-8") as pgn_file:
        game_count = 0

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
        
            if len(list(game.mainline_moves())) == 0:
                continue

            for move in game.mainline_moves():
                position = board_to_tensor(board)
                board.push(move)

                next_move = move.uci()
                chunk.append((position, next_move))

                if len(chunk) >= chunk_size:
                    save_chunk_to_file(chunk, output_file)
                    chunk = []
            game_count += 1

            if game_count % 1000 == 0:
                print(f"🔥 Processed {game_count} games...")
            if game_count == 1_000_001:
                break
        if chunk:
            save_chunk_to_file(chunk, output_file)

    print(f"✅ Extraction complete! Processed a total of {game_count} games.")


def save_chunk_to_file(chunk, output_file):
    """
    Append chunk to file incrementally using pickle.
    """
    with open(output_file, 'ab') as f:
        pickle.dump(chunk, f)

def load_data_in_chunks(file_path):
    """
    Load data from file in chunks using pickle.
    """
    with open(file_path, 'rb') as f:
        while True:
            try:
                chunk = pickle.load(f)
                yield chunk
            except EOFError:
                break

def decompress_and_filter_pgn(file_path, output_path, min_elo=1000, max_elo=2000):
    """
    Decompresses a PGN .zst file and filters games with player Elo ratings between min_elo and max_elo.
    """
    with open(file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed_file) as reader:
            raw_data = reader.read().decode('utf-8')  # ✅ Read entire decompressed data
            lines = raw_data.split("\n")  # ✅ Split into lines manually

    with open(output_path, 'w', encoding='utf-8') as filtered_pgn:
        buffer = []
        inside_game = False
        valid_game = False

        for line in lines:
            line = line.strip()
            
            # Start of a new game
            if line.startswith("[Event "): 
                if valid_game and buffer:
                    filtered_pgn.write("\n".join(buffer) + "\n\n")  # ✅ Save previous valid game
                buffer = []
                inside_game = True
                valid_game = False  # Reset for the new game

            buffer.append(line)

            # Check for Elo ratings
            if line.startswith("[WhiteElo") or line.startswith("[BlackElo"):
                elo_match = re.findall(r'\"(\d+)\"', line)
                if elo_match:
                    elo = int(elo_match[0])
                    if min_elo <= elo <= max_elo:
                        valid_game = True  # ✅ Mark game as valid

        # Save the last game if it's valid
        if valid_game and buffer:
            filtered_pgn.write("\n".join(buffer) + "\n\n")

    print(f"✅ Filtered PGN saved to {output_path}")

# File paths
#zst_file_path = r'C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\chess.ai\Data\lichess_db_standard_rated_2016-03.pgn.zst'
pgn_file_path = r'C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\chess.ai\Data\1000-2000_elo_games.pgn'
output_file = r'1000-2000_elo_games.pkl'

#decompress_and_filter_pgn(zst_file_path, output_file)

# with open(pgn_file_path, 'r', encoding="utf-8") as pgn_file:
#     for _ in range(5):  # Read the first 5 games
#         raw_game = "".join([pgn_file.readline() for _ in range(20)])  # Read first 20 lines
#         print("🔍 Raw PGN Content:\n", raw_game)


# Extract data in chunks
extract_data_in_chunks(pgn_file_path=pgn_file_path, output_file=output_file)

# # Example of loading data
# print("Loading data from saved file:")
# for chunk in load_data_in_chunks(output_file):
#     for position, next_move in chunk:
#         print(position.shape, next_move)