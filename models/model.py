from facenet_pytorch import MTCNN, InceptionResnetV1
from config import device

# Initialize MTCNN and FaceNet model
def initialize_models():

    '''
    model : https://github.com/ipazc/mtcnn
    key Reason - 
        1. Dedicated Model for face Recognition 
        2. High Efficiency and Accuracy
        3. Hierarchical Feature Extraction for Extracting the Embedding Point of the face 
    '''

    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, model
