from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions

app = FastAPI(title="Dog Breed Classifier")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained ResNet50V2 model
print("Loading model...")
model = ResNet50V2(weights='imagenet')
print("Model loaded successfully.")

# ImageNet dog breed indices (151 to 268 inclusive)
DOG_START_INDEX = 151
DOG_END_INDEX = 268

def process_image_opencv(image_bytes: bytes) -> np.ndarray:
    # Use OpenCV to read the image bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    
    # Resize to 224x224 for ResNet50V2
    img_resized = cv2.resize(img, (224, 224))
    
    # OpenCV loads in BGR, ResNet expects RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Expand dims to match batch size (1, 224, 224, 3)
    img_expanded = np.expand_dims(img_rgb, axis=0)
    
    # Preprocess for ResNetV2
    img_preprocessed = preprocess_input(np.array(img_expanded, dtype=np.float32))
    return img_preprocessed

@app.post("/predict")
async def predict_dog_breed(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        contents = await file.read()
        processed_img = process_image_opencv(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    # Run inference
    preds = model.predict(processed_img)
    
    # Check if the top prediction overall is a dog
    top_pred_index = np.argmax(preds[0])
    
    is_dog = DOG_START_INDEX <= top_pred_index <= DOG_END_INDEX
    
    if not is_dog:
        # Decode the top non-dog prediction just to give feedback
        decoded_top = decode_predictions(preds, top=1)[0][0]
        return {
            "is_dog": False,
            "message": f"This looks more like a {decoded_top[1].replace('_', ' ')} than a dog.",
            "predictions": []
        }
    
    # Filter only dog predictions for the output
    # Create a mask for dogs
    dog_preds = preds[0].copy()
    mask = np.zeros_like(dog_preds)
    mask[DOG_START_INDEX:DOG_END_INDEX + 1] = 1
    dog_preds = dog_preds * mask
    
    # Get top 5 dog predictions
    top_5_indices = dog_preds.argsort()[-5:][::-1]
    
    # Decode them
    # Create a fake prediction array to use decode_predictions
    fake_preds = np.zeros((1, 1000))
    for i, idx in enumerate(top_5_indices):
        fake_preds[0, idx] = dog_preds[idx]
        
    decoded_dogs = decode_predictions(fake_preds, top=5)[0]
    
    results = []
    for _id, label, prob in decoded_dogs:
        if prob > 0: # only include if prob > 0
            results.append({
                "breed": label.replace('_', ' ').title(),
                "confidence": float(prob)
            })
            
    return {
        "is_dog": True,
        "message": "Dog detected successfully.",
        "predictions": results
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
