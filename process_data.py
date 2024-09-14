import json
import numpy as np
import torch
import zstandard as zstd
import io

INPUT_FILE = 'lichess_db_eval.jsonl.zst'  # https://database.lichess.org/#evals
OUTPUT_FILE = 'data.pt'
MAX_SAMPLES = 50_000

piece_to_index = {
    'P': 0,
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11
}


def fen_to_tensor(fen):
    total_channels = 12 + 1 + 4 + 1  # 18 channels
    board_tensor = np.zeros((total_channels, 8, 8), dtype=np.float32)

    fen_parts = fen.strip().split(' ')

    board_fen = fen_parts[0]
    active_color = fen_parts[1]
    castling_rights = fen_parts[2]
    en_passant_square = fen_parts[3]

    rows = board_fen.split('/')

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            elif char in piece_to_index:
                piece_idx = piece_to_index[char]
                board_tensor[piece_idx, row_idx, col_idx] = 1.0
                col_idx += 1
            else:
                raise ValueError(f"Invalid character '{char}' in FEN string.")

    # Plane 12: Side to move
    if active_color == 'w':
        board_tensor[12, :, :] = 1.0
    elif active_color == 'b':
        board_tensor[12, :, :] = 0.0
    else:
        raise ValueError(f"Invalid active color '{
                         active_color}' in FEN string.")

    # Planes 13-16: Castling rights
    # Plane 13: White can castle kingside (K)
    # Plane 14: White can castle queenside (Q)
    # Plane 15: Black can castle kingside (k)
    # Plane 16: Black can castle queenside (q)
    if castling_rights != '-':
        if 'K' in castling_rights:
            board_tensor[13, :, :] = 1.0
        if 'Q' in castling_rights:
            board_tensor[14, :, :] = 1.0
        if 'k' in castling_rights:
            board_tensor[15, :, :] = 1.0
        if 'q' in castling_rights:
            board_tensor[16, :, :] = 1.0

    # Plane 17: En passant target square
    if en_passant_square != '-':
        file = en_passant_square[0]
        rank = en_passant_square[1]
        col = ord(file) - ord('a')
        row = 8 - int(rank)
        board_tensor[17, row, col] = 1.0

    return board_tensor


def process_data():
    positions = []
    evaluations = []

    dctx = zstd.ZstdDecompressor()

    with open(INPUT_FILE, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line_num, line in enumerate(text_stream, 1):
                if line_num % 100000 == 0:
                    print(f"Processed {line_num} lines")

                try:
                    line = line.strip()
                    data = json.loads(line)
                    fen = data['fen']
                    evals = data['evals']

                    if not evals:
                        continue

                    eval_info = evals[0]['pvs'][0]

                    if 'cp' in eval_info:
                        # Normalize centipawn values to pawn units
                        target = eval_info['cp'] / 100.0
                    elif 'mate' in eval_info:
                        # Use a large value for mate scores
                        mate_score = eval_info['mate']
                        target = 1000.0 if mate_score > 0 else -1000.0
                    else:
                        continue

                    positions.append(fen)
                    evaluations.append(target)

                    # Check if we've reached the desired number of samples
                    if len(positions) >= MAX_SAMPLES:
                        print(
                            f"Reached {MAX_SAMPLES} samples. Stopping data collection.")
                        break

                except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                    print(f"Skipping line {line_num} due to error: {e}")
                    continue

    print(f"Total positions collected: {len(positions)}")

    # Convert FEN strings to tensors
    input_tensors = []
    for idx, fen in enumerate(positions):
        if idx % 10000 == 0 and idx > 0:
            print(f"Converted {idx} / {MAX_SAMPLES} positions to tensors")
        try:
            tensor = fen_to_tensor(fen)
            input_tensors.append(tensor)
        except ValueError as e:
            print(f"Skipping position {idx} due to error: {e}")
            continue

    print(f"Total tensors created: {len(input_tensors)}")

    # Ensure that input_tensors and evaluations have the same length
    min_length = min(len(input_tensors), len(evaluations))
    input_tensors = np.array(input_tensors[:min_length], dtype=np.float32)
    target_tensors = np.array(evaluations[:min_length], dtype=np.float32)

    input_tensors = torch.from_numpy(input_tensors)
    target_tensors = torch.from_numpy(target_tensors)

    # Save only the collected samples
    torch.save({'inputs': input_tensors,
                'targets': target_tensors}, OUTPUT_FILE)
    print(f"Data saved to '{OUTPUT_FILE}' with {min_length} samples.")


if __name__ == "__main__":
    process_data()
