import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import chess
import chess.engine
import pygame.image
import numpy as np
import random

pygame.init()

class BoardCNN(nn.Module):
    def __init__(self, cnn_output_dim=256):
        super(BoardCNN, self).__init__()
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256 * 8 * 8, cnn_output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class MoveTransformer(nn.Module):
    def __init__(self, transformer_dim=256, num_heads=8, num_layers=4):
        super(MoveTransformer, self).__init__()
        self.embedding = nn.Linear(transformer_dim, transformer_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x) 
        x = self.transformer(x)
        return x
    
class HybridChessModel(nn.Module):
    def __init__(self, cnn_output_dim=256, transformer_dim=256, num_classes=20480):
        super(HybridChessModel, self).__init__()
        self.cnn = BoardCNN(cnn_output_dim)
        self.transformer = MoveTransformer(transformer_dim=cnn_output_dim)
        self.fc = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        print("x.shape:", x.shape)
        batch_size, seq_len, channels, height, width = x.shape

        x = x.view(batch_size * seq_len, channels, height, width)

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, seq_len, -1)

        transformer_output = self.transformer(cnn_output)
        final_features = transformer_output[:, -1, :]

        logits = self.fc(final_features)
        return logits

PIECES = {
    'P': pygame.image.load("pieces/wP.png"),  # White Pawn
    'R': pygame.image.load("pieces/wR.png"),  # White Rook
    'N': pygame.image.load("pieces/wN.png"),  # White Knight
    'B': pygame.image.load("pieces/wB.png"),  # White Bishop
    'Q': pygame.image.load("pieces/wQ.png"),  # White Queen
    'K': pygame.image.load("pieces/wK.png"),  # White King
    'p': pygame.image.load("pieces/p.png"),   # Black Pawn
    'r': pygame.image.load("pieces/r.png"),   # Black Rook
    'n': pygame.image.load("pieces/n.png"),   # Black Knight
    'b': pygame.image.load("pieces/b.png"),   # Black Bishop
    'q': pygame.image.load("pieces/q.png"),   # Black Queen
    'k': pygame.image.load("pieces/k.png"),   # Black King
}

# Game constants
WIDTH, HEIGHT = 512*2, 512*2
SQUARE_SIZE = WIDTH // 8
BOARD_COLOR_LIGHT = (240, 217, 181)
BOARD_COLOR_DARK = (181, 136, 99)

# Initialize pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Chess")

# Chess game logic
board = chess.Board()
dragging_piece = None
dragging_rect = None
start_square = None

PROMOTION_INVERSE_MAP = {
    0: None,
    1: chess.QUEEN,
    2: chess.ROOK,
    3: chess.BISHOP,
    4: chess.KNIGHT,
}

def integer_to_move(target_int):
    promotion_flag = target_int % 5
    move_code = target_int // 5
    from_square = move_code // 64
    to_square = move_code % 64
    promotion_piece = PROMOTION_INVERSE_MAP[promotion_flag]
    return chess.Move(from_square, to_square, promotion=promotion_piece)

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
            channel = p_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            row, col = divmod(square, 8)
            tensor[row, col, channel] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[0, 4, 12] = 1
        tensor[0, 7, 12] = 1
    else:
        tensor[0, 4, 12] = 0
        tensor[0, 7, 12] = 0

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[0, 4, 12] = 1
        tensor[0, 0, 12] = 1  
    else:
        tensor[0, 4, 12] = 0
        tensor[0, 0, 12] = 0

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[7, 4, 13] = 1
        tensor[7, 7, 13] = 1 
    else:
        tensor[7, 4, 13] = 0
        tensor[7, 7, 13] = 0

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[7, 4, 13] = 1  
        tensor[7, 0, 13] = 1  
    else:
        tensor[7, 4, 13] = 0
        tensor[7, 0, 13] = 0


    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[7, 4, 13] = 1
        tensor[7, 7, 13] = 1
    else:
        tensor[7, 4, 13] = 0
        tensor[7, 7, 13] = 0

    tensor = np.transpose(tensor, (2, 0, 1))
    return tensor

def draw_board():
    """Draw the chessboard with rank and file coordinates."""
    font = pygame.font.Font(None, 24)

    for row in range(8):
        for col in range(8):
            color = BOARD_COLOR_LIGHT if (row + col) % 2 == 0 else BOARD_COLOR_DARK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            if row == 7:
                file_label = font.render(chr(ord('a') + col), True, (0, 0, 0) if color == BOARD_COLOR_LIGHT else (255, 255, 255))
                screen.blit(file_label, (col * SQUARE_SIZE + 5, HEIGHT - 20))

            if col == 0:
                rank_label = font.render(str(8 - row), True, (0, 0, 0) if color == BOARD_COLOR_LIGHT else (255, 255, 255))
                screen.blit(rank_label, (5, row * SQUARE_SIZE + 5))


def draw_pieces():
    """Draw pieces on the board."""
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece:
                img = PIECES[piece.symbol()]
                img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(img, (col * SQUARE_SIZE, row * SQUARE_SIZE))

def get_square_under_mouse(pos):
    """Convert pixel position to chess square."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return chess.square(col, 7 - row)

def handle_promotion_dialog():
    """Handle promotion choice."""
    font = pygame.font.Font(None, 36)
    choices = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
    text_surfaces = {key: font.render(key.upper(), True, (255, 255, 255)) for key in choices}
    choice_rects = {}

    for i, key in enumerate(choices):
        rect = pygame.Rect(200 + i * 60, HEIGHT // 2 - 30, 50, 50)
        pygame.draw.rect(screen, (0, 0, 0), rect)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        screen.blit(text_surfaces[key], (rect.x + 10, rect.y + 10))
        choice_rects[key] = rect

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for key, rect in choice_rects.items():
                    if rect.collidepoint(event.pos):
                        return choices[key]

def tensor_to_board_14x8x8(tensor):
    """
    Converts a 14x8x8 NumPy array to a chess.Board object.
    """
    board = chess.Board()
    board.clear()

    piece_map = {
        0: chess.PAWN,
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
        5: chess.KING
    }

    for row in range(8):
        for col in range(8):
            for channel in range(12):
                if tensor[channel, row, col] == 1:
                    piece_type = piece_map[channel % 6]
                    color = chess.WHITE if channel < 6 else chess.BLACK
                    square = chess.square(col, 7 - row)

                    board.set_piece_at(square, chess.Piece(piece_type, color))

    return board

def main():
    global dragging_piece, dragging_rect, start_square

    # You can let the user choose here (via input or a GUI dialog)
    # For simplicity, we'll hard-code it. To have AI play as White, set to chess.WHITE.
    ai_color = chess.BLACK  
    user_color = chess.WHITE if ai_color == chess.BLACK else chess.BLACK

    # Load the model as before.
    model = HybridChessModel()
    model.load_state_dict(torch.load(
        r"D:\AI Projects\chess.ai\models\GM-7(best_val).pth", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print('Model loaded successfully!')

    running = True
    game_over = False
    game_tensors = []
    is_start_state = True
    seq_len = 10

    # Main game loop
    while running:
        screen.fill((0, 0, 0))
        draw_board()
        draw_pieces()

        if is_start_state:
            for _ in range(seq_len):
                game_tensors.append(board_to_tensor(board))
            print('game tensors shape at initial:', np.array(game_tensors).shape)
            is_start_state = False
            continue

        # Check if the game is over
        if board.is_game_over() and not game_over:
            game_over = True
            result = board.result()
            if result == "1-0":
                game_over_message = "White Wins!"
            elif result == "0-1":
                game_over_message = "Black Wins!"
            else:
                game_over_message = "Draw!"

        if game_over:
            font = pygame.font.Font(None, 74)
            text = font.render(game_over_message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            time.sleep(5)
            break

        # AI's Move
        if board.turn == ai_color and not game_over:
            print("AI is making a move...")
            np_game = np.array(game_tensors)
            print(np_game.shape)
            input_tensor = torch.tensor(np_game, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                top5_indices = torch.topk(logits, 50).indices.squeeze(0).tolist()

            predicted_move = None
            move_num = 1
            for move_index in top5_indices:
                move_candidate = integer_to_move(move_index)
                if move_candidate in board.legal_moves:
                    predicted_move = move_candidate
                    print(f"AI ({'White' if ai_color == chess.WHITE else 'Black'}) selects top move({move_num}): {predicted_move.uci()}")
                    break
                move_num += 1

            if not predicted_move:
                predicted_move = random.choice(list(board.legal_moves))

            board.push(predicted_move)
            game_tensors = game_tensors[1:] + [board_to_tensor(board)]
            # No need to toggle a separate flag: board.turn is updated automatically

        # Event handling for the human player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Only allow human interaction when it is the user's turn.
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over and board.turn == user_color:
                start_square = get_square_under_mouse(event.pos)
                piece = board.piece_at(start_square)
                if piece:
                    dragging_piece = PIECES[piece.symbol()]
                    dragging_rect = pygame.Rect(
                        event.pos[0] - SQUARE_SIZE // 2,
                        event.pos[1] - SQUARE_SIZE // 2,
                        SQUARE_SIZE, SQUARE_SIZE
                    )

            elif event.type == pygame.MOUSEMOTION and dragging_piece:
                dragging_rect.topleft = (event.pos[0] - SQUARE_SIZE // 2, event.pos[1] - SQUARE_SIZE // 2)

            elif event.type == pygame.MOUSEBUTTONUP and not game_over and board.turn == user_color:
                if dragging_piece:
                    end_square = get_square_under_mouse(event.pos)
                    move = chess.Move(start_square, end_square)

                    piece = board.piece_at(start_square)  # Retrieve the piece safely
                    if piece is not None and piece.piece_type == chess.PAWN and (
                        end_square in chess.SquareSet(chess.BB_RANK_8) or end_square in chess.SquareSet(chess.BB_RANK_1)
                    ):
                        promotion_piece = handle_promotion_dialog()
                        move = chess.Move(start_square, end_square, promotion=promotion_piece)

                    if move in board.legal_moves:
                        board.push(move)
                        game_tensors = game_tensors[1:] + [board_to_tensor(board)]
                    else:
                        print("Illegal move attempted:", move.uci())
                        
                    dragging_piece = None
                    dragging_rect = None

        if dragging_piece and dragging_rect:
            screen.blit(dragging_piece, dragging_rect)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()