import os
import torch
from .extract_embeddings import extract_video_embeddings

def save_training_embeddings(video_data_folder, model, mtcnn):

    '''
    Parameters : 
        video data folder : Input Person Face Video Data
        model : Model to be used for extracting embeddings
        mtcnn : MTCNN Model to be used for face detection

    Returns:
        print the face is detected and parse to embedded.pt with name of person
    '''

    for video_file in os.listdir(video_data_folder):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_data_folder, video_file)
        print(f"Processing {video_name}...")
        
        embedding = extract_video_embeddings(video_path, model, mtcnn)
        if embedding is not None:
            torch.save(embedding, os.path.join(video_data_folder, f"{video_name}.pt"))
        else:
            print(f"No faces detected in {video_name}.")

def save_testing_embeddings(test_video_path, model, mtcnn):

    '''
    Parameters :
        test_video_path : Testing Video Feed
        model : Model to be used for extracting embeddings
        mtcnn : MTCNN Model to be used for face detection

    Returns :
        parse the video and create the embedding for the face in the test.pt
    '''

    embedding = extract_video_embeddings(test_video_path, model, mtcnn)
    if embedding is not None:
        torch.save(embedding, "test.pt")
        print("Testing embeddings saved as test.pt.")
    else:
        print("No faces detected in the test video.")
