import pickle
import torch
import chess
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import chess.svg
from IPython.display import display, SVG

torch.set_float32_matmul_precision('high')

file_path = r'C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\chess.ai\Data\1000-2000_elo_games.pkl'
#file_path = r"D:\AI Projects\Chess.ai\grandmastergames.pkl"

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

class SmallChessModel(nn.Module):
    def __init__(self, num_classes=20480):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class HybridChessModel(nn.Module):
    def __init__(self, cnn_output_dim=256, transformer_dim=256, num_classes=20480):
        super(HybridChessModel, self).__init__()
        self.cnn = BoardCNN(cnn_output_dim)
        self.transformer = MoveTransformer(transformer_dim=cnn_output_dim)
        self.fc = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape

        x = x.view(batch_size * seq_len, channels, height, width)

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, seq_len, -1)

        transformer_output = self.transformer(cnn_output)
        final_features = transformer_output[:, -1, :]

        logits = self.fc(final_features)
        return logits

def display_chessboard(board):
    """Render the chessboard as an SVG."""
    if not isinstance(board, chess.Board): 
        raise ValueError("Expected a chess.Board object but got", type(board))

    svg = chess.svg.board(board, size=400)
    display(SVG(svg)) 

def load_data_from_pickle(file_path, num_to_load):
    data = []
    with open(file_path, 'rb') as f:
        count = 0
        while count <= num_to_load - 1:
            try:
                chunk = pickle.load(f)
                data.extend(chunk)
                count += 1
            except EOFError:
                break
    return data

def preprocess_data_for_sequences(data, seq_len=10):
    inputs, targets = [], []
    for i in range(len(data) - seq_len):
        sequence = [torch.tensor(data[j][0], dtype=torch.float32).permute(2, 0, 1)  # (14, 8, 8)
                    for j in range(i, i + seq_len)]
        
        move = data[i + seq_len][1] 
        try:
            from_square = chess.SQUARE_NAMES.index(move[:2])
            to_square = chess.SQUARE_NAMES.index(move[2:4])
            target = from_square * 64 + to_square
        except ValueError:
            print(f"Skipping invalid move: {move}")
            continue

        inputs.append(torch.stack(sequence))
        targets.append(target)

    return torch.stack(inputs), torch.tensor(targets, dtype=torch.long)

PROMOTION_MAP = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

PROMOTION_INVERSE_MAP = {
    0: None,
    1: chess.QUEEN,
    2: chess.ROOK,
    3: chess.BISHOP,
    4: chess.KNIGHT,
}

def move_to_integer(move):
    move = chess.Move.from_uci(move)
    from_square = move.from_square
    to_square = move.to_square
    promotion_flag = PROMOTION_MAP[move.promotion] 
    return (from_square * 64 + to_square) * 5 + promotion_flag

def integer_to_move(target_int):
    promotion_flag = target_int % 5
    move_code = target_int // 5
    from_square = move_code // 64
    to_square = move_code % 64
    promotion_piece = PROMOTION_INVERSE_MAP[promotion_flag]
    return chess.Move(from_square, to_square, promotion=promotion_piece)

def load_data_in_chunks(file_path):
    with open(file_path, 'rb') as f:
        while True:
            try:
                chunk = pickle.load(f)
                yield chunk
            except EOFError:
                break

def is_inital_board(board):
    INITIAL_FEN = "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w - - 0 1"
    return board.fen() == INITIAL_FEN

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

def generate_batches_in_sequence(file_path, batch_size, seq_len=10, start_chunk=0, end_chunk=None):
    buffer_x = []
    buffer_y = []

    seq_x = []
    count = 0
    curr_count = 0
    prev_count = 0
    for chunk_idx, chunk in enumerate(load_data_in_chunks(file_path)):
        if chunk_idx < start_chunk:
            continue
        if end_chunk is not None and chunk_idx >= end_chunk:
            break

        for (position, next_move) in chunk:
            count += 1
            position_14x8x8 = np.transpose(position, (2, 0, 1))
            board = tensor_to_board_14x8x8(position_14x8x8)

            if is_inital_board(board):
                prev_count = curr_count
                curr_count = count
                seq_x = [position_14x8x8.copy() for _ in range(seq_len)]
            else:
                if not seq_x:
                    seq_x.append(position_14x8x8.copy())
                else:
                    seq_x = seq_x[1:] + [position_14x8x8.copy()]

            if len(seq_x) == seq_len:
                buffer_x.append(seq_x.copy())
                buffer_y.append(move_to_integer(next_move))

            if len(buffer_x) == batch_size:
                x_batch = np.array(buffer_x, dtype=np.float32)
                y_batch = np.array(buffer_y, dtype=np.int64)
                yield (x_batch, y_batch)
                buffer_x = []
                buffer_y = []

    if buffer_x:
        x_batch = np.array(buffer_x, dtype=np.float32)
        y_batch = np.array(buffer_y, dtype=np.int64)
        yield (x_batch, y_batch)


def count_total_batches(file_path, batch_size):
    total_positions = 0

    for chunk in load_data_in_chunks(file_path):
        total_positions += len(chunk)

    full_batches, leftover = divmod(total_positions, batch_size)
    if leftover > 0:
        return full_batches + 1
    else:
        return full_batches

def top_k_accuracy(logits, targets, k=5):

    _, topk_indices = torch.topk(logits, k, dim=1)
    correct = topk_indices.eq(targets.view(-1, 1))

    topk_correct = correct.any(dim=1).float().sum().item()
    batch_size = targets.size(0)
    topk_acc = topk_correct / batch_size

    return topk_acc * 100

def get_batch(generator):
    try:
        X_np, Y_np = next(generator)
    except StopIteration:
        generator = generate_batches_in_sequence(file_path, batch_size=batch_size)
        X_np, Y_np = next(generator)

    X = torch.from_numpy(X_np).to('cuda')
    Y = torch.tensor(Y_np, dtype=torch.long).to('cuda')

    return X, Y, generator

def debug_tensor_positions(tensor):
    """Prints the tensor contents for debugging"""
    piece_map = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

    for channel in range(12):
        for row in range(8):
            for col in range(8):
                if tensor[channel, row, col] == 1:
                    piece = piece_map[channel]
                    print(f"Piece {piece} at ({row}, {col})")

num_steps = 1_000_000 * 2
data_length = 83175752
data_chunks = 83176
num_batches = 649810

print(f'Number of positions {data_length}')
print(f'Number of chunks {data_chunks}')
print("Number of batches:", num_batches)

model = HybridChessModel()
model.load_state_dict(torch.load(r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\chess.ai\models\GM-7(best_val).pth", weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_split_idx = data_chunks - (int)(data_chunks * 0.99975)
val_split_idx = ((int)(data_chunks * 0.00025))
best_val_loss = float('inf')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

batch_size = 32
train_generator = generate_batches_in_sequence(file_path, batch_size=batch_size, start_chunk=train_split_idx, end_chunk=None)
val_generator = generate_batches_in_sequence(file_path, batch_size, start_chunk=0, end_chunk=val_split_idx)

print('Starting training...\n')

# with open(file_path, "rb") as f:
#     sample_data = pickle.load(f)
#     print(sample_data[:1])  # See first 5 moves

# board = chess.Board()

# for i in range(10):
#     print(f"📝 Sample {i+1}")
#     x_sample, y_sample, train_generator = get_batch(train_generator)
#     print("Position Tensor Shape:", x_sample.shape)
#     print("Encoded Moves:", y_sample.shape)

#     uci_moves = [integer_to_move(move).uci() for move in y_sample]

#     #print("Decoded Moves:", uci_moves)
#     #print("Legal Moves in This Position:", [m.uci() for m in board.legal_moves])
#     print("-" * 50)
#     input()


#import sys; sys.exit()


start_time = time.time()

for step in range(num_steps):
    #train_start_time = time.time()
    X, Y, train_generator = get_batch(train_generator)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(X)
        loss = criterion(logits, Y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % 10_000 == 0 or step == num_steps - 1:
        model.eval()
        total_val_loss = 0 
        val_steps = 200
        with torch.no_grad():
            for _ in range(val_steps):
                X_val, Y_val, val_generator = get_batch(val_generator)
                val_logits = model(X_val)
                val_loss = criterion(val_logits, Y_val.view(-1))
                total_val_loss += val_loss.item()
        top1 = top_k_accuracy(val_logits, Y_val, k=1)
        top5 = top_k_accuracy(val_logits, Y_val, k=5)
        top10 = top_k_accuracy(val_logits, Y_val, k=10)
        total_val_loss /= val_steps
        scheduler.step(total_val_loss)
        #torch.cuda.synchronize()
        end_time = time.time()
        dt = end_time - start_time
        print(f"Step {step}/{num_steps}, Train loss: {loss.item():.8f}, Val loss: {total_val_loss:.8f}, Learning rate: {optimizer.param_groups[0]['lr']}, top1: {top1:.3f}, top5: {top5:.3f}, top10: {top10:.3f}, dt: {dt:.5f}")
        start_time = time.time()
        model.train()

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            print("Saving best val loss model...")
            torch.save(model.state_dict(), "GM-8(best_val).pth")
            print("Saved!")
    # train_end_time = time.time()
    # train_dt = train_end_time - train_start_time
    # print(f'Train loss: {loss.item():.8f}, dt: {train_dt:.5f}')
torch.save(model, "GM-8(full_model).pth")
print("Model saved to my_chess_model.pth")