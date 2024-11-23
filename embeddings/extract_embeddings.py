import cv2
import torch
from config import device

def extract_video_embeddings(video_path, model, mtcnn):
    
    '''
    Parameter :    
        video_path (str) : path to the video file
        model (nn.Module) : model to use for extracting embeddings
        mtcnn (MTCNN) : MTCNN model to use for face detection

    Returns :
        Embedding of the face from the Video
    '''

    cap = cv2.VideoCapture(video_path)
    embeddings = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using MultiTask Cascaded Convolutional Network
        faces, _ = mtcnn.detect(frame)
        if faces is None:
            continue

        # Process each detected face
        frame_height, frame_width = frame.shape[:2]
        for (x1, y1, x2, y2) in faces:
            x1, y1, x2, y2 = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, frame_width)), int(min(y2, frame_height))
            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            face_tensor = face_tensor.to(device)

            # Generate face embedding
            embedding = model(face_tensor).detach().cpu()
            embeddings.append(embedding)
    
    cap.release()

    if embeddings:
        return torch.mean(torch.stack(embeddings), dim=0)
    return None
