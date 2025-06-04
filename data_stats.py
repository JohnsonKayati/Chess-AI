import pickle

file_path = r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\chess.ai\Data\grandmastergames.pkl"

with open(file_path, "rb") as f:
    total_positions = 0
    total_chunks = 0

    while True:
        try:
            chunk = pickle.load(f)  # Load one chunk at a time
            total_positions += len(chunk)  # Add the number of positions in the chunk
            total_chunks += 1
            #print(f"Loaded chunk {total_chunks}: {len(chunk)} positions")
        except EOFError:
            break

print(f"Total positions: {total_positions}, Total chunks: {total_chunks}")
