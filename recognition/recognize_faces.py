import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

def recognize_faces(test_embedding_path, training_embeddings_folder, threshold=0.5, log_file="log.txt"):
    
    """
    Recognize faces from test embeddings by comparing them to training embeddings using cosine similarity.

    Parameters:
        test_embedding_path (str): Path to the test embeddings .pt file.
        training_embeddings_folder (str): Folder containing training embeddings (.pt files).
        threshold (float): Similarity threshold to classify as a match.
        log_file (str): Path to the log file where results will be stored.

    Returns:
        dict: Results containing detected names and their confidence scores.
    """
    # Load testing embeddings
    test_embeddings = torch.load(test_embedding_path)
    
    # Ensure that test_embeddings is a list of tensors
    if isinstance(test_embeddings, dict):
        print("Error: test_embeddings should be a list of tensors, not a dictionary.")
        return []

    # Load training embeddings
    training_embeddings = {}
    for file in os.listdir(training_embeddings_folder):
        if file.endswith(".pt"):
            name = os.path.splitext(file)[0]
            train_embedding = torch.load(os.path.join(training_embeddings_folder, file))
            
            # Ensure the loaded embedding is a tensor
            if isinstance(train_embedding, torch.Tensor):
                training_embeddings[name] = train_embedding
            else:
                print(f"Warning: {file} is not a valid embedding tensor.")

    results = []  # To store results for each test embedding
    batch_size = 10  # Process every 10 frames
    batch_results = []  # To store results of the current batch

    with open(log_file, 'a') as log:  # Open the log file in append mode
        # Process embeddings in batches of 10
        for i in range(0, len(test_embeddings), batch_size):
            # Process next 10 embeddings
            batch_embeddings = test_embeddings[i:i+batch_size]
            batch_results.clear()  # Clear previous batch results

            for j, test_embedding in enumerate(batch_embeddings):
                best_match = None
                best_score = -1
                
                # Ensure test_embedding is a tensor
                if not isinstance(test_embedding, torch.Tensor):
                    print(f"Warning: Test embedding {i+j} is not a tensor.")
                    continue
                
                # Flatten the embeddings to 1D (important for cosine similarity)
                test_embedding = test_embedding.view(-1).numpy()  # Flatten and convert to numpy
                
                for name, train_embedding in training_embeddings.items():
                    # Flatten and convert the train_embedding to 1D numpy array
                    train_embedding = train_embedding.view(-1).numpy()  # Flatten and convert to numpy
                    
                    # Compute cosine similarity
                    score = cosine_similarity([test_embedding], [train_embedding])[0][0]
                    
                    # Update best match if the score is higher
                    if score > best_score:
                        best_score = score
                        best_match = name
                
                # Append results
                if best_score > threshold:
                    result = {"name": best_match, "score": best_score}
                    batch_results.append(f"Face {i+j+1}: Detected {best_match} with confidence {best_score:.2f}")
                else:
                    result = {"name": "Unknown", "score": best_score}
                    batch_results.append(f"Face {i+j+1}: Alert - Not in data (best score: {best_score:.2f})")
            
            # After every batch, log the results
            log.write("\n".join(batch_results) + "\n")
            log.write("------ End of Batch ------\n")
            print(f"Processed batch {i//batch_size + 1} of {len(test_embeddings)//batch_size + 1}")

    return results