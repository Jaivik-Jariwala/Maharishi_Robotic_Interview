from models.model import initialize_models
from embeddings.save_embeddings import save_training_embeddings, save_testing_embeddings
from recognition.recognize_faces import recognize_faces
from config import video_data, test_data, training_Embedding

# Initialize models
mtcnn, model = initialize_models()

# run to create and Save embeddings
save_training_embeddings(video_data, model, mtcnn)
save_testing_embeddings(test_data, model, mtcnn)

# Perform recognition and create log file
results = recognize_faces("test.pt", training_Embedding, threshold=0.3, log_file="D:/Internview/Maharishi/output/log.txt")

