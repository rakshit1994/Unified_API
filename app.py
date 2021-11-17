from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import Google_PubSub.pubsub as pubsub
from PIL import Image
from io import BytesIO
import ML_Model.cnn_model as model
import shutil
import uvicorn

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

@app.get("/", response_class=HTMLResponse)
async def render_home(request: Request):
    return templates.TemplateResponse('index.html', context={"request": request})

@app.get("/results", response_class=HTMLResponse)
async def render_home(request: Request):
    return templates.TemplateResponse('results.html', context={"request": request})

@app.get("/fetch")
async def fetch_results():
    return pubsub.pull()


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    imgGray = image.convert('L')
    pubsub_id = model.predict_img(imgGray)
    return pubsub_id

if __name__=="__main__":
    uvicorn.run(app,port='8000',host='0.0.0.0')