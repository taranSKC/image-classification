from fastapi import FastAPI, UploadFile, File
from PIL import Image
from model import model
import torchvision.transforms as transforms
import torch
import json
import uvicorn


app = FastAPI()


@app.on_event("startup")
async def load_model():
    global mymodel
    mymodel = model
    mymodel.load_state_dict(torch.load('taran_weights.pth'))
    mymodel.eval()


@app.post("/classify")
async def resize_image(file: UploadFile = File(...)):
    # Read the uploaded image

    image = Image.open(file.file).convert('RGB')
    input_data = preprocess(image)
    input_data = input_data.unsqueeze(0)

    output = mymodel(input_data)

    output = torch.softmax(output, dim=1)
    values, indices = torch.topk(output, k=5)
    print(indices.shape)
    print(values.shape, "values")
    zipped_output = list(zip(values[0].tolist(), indices[0].tolist()))

    with open('all_classes.json') as f:
        all_classes = json.load(f)
    print(zipped_output)
    output_send = {
        idx + 1:    all_classes[index] + " - " + str(round(percentage*100, 2)) + "%" for idx, (percentage, index) in enumerate(zipped_output)
    }

    return {"result": output_send}


preprocess = transforms.Compose([
    # transforms.Resize(224),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# uvicorn api:app --reload
