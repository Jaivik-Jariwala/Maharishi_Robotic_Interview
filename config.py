import torch

# Config for device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data paths

video_data = "D:/Internview/Maharishi/video data"
test_data = "D:/Internview/Maharishi/test data/Testing_video.mp4"
training_Embedding = "D:/Internview/Maharishi/embedding_pt/"