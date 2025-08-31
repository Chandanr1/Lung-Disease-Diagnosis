import torch
from xray.ml.model.arch import Net
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile

app=FastAPI()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
model.load_state_dict(torch.load('xray_model.pth',map_location=device))
model.eval()

transform =transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

label_map={0:'NORMAL',1:'PNEUMONIA'}
@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image=Image.open(file.file).convert('RGB')
    input_tensor=transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output=model(input_tensor)
        prediction_index=torch.argmax(output,dim=1).item()
        predicted_label=label_map.get(prediction_index,'unknown')
    return {"prediction_index":prediction_index,"prediction_label":predicted_label}