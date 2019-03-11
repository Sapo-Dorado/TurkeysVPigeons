from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
from flask import render_template
import aiohttp
import uvicorn

from fastai.vision import open_image, load_learner
from pathlib import Path
from io import BytesIO
import numpy as np

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def get_prediction(bytes, request):
    img = open_image(BytesIO(bytes))
    _,_,losses = learn.predict(img)
    pred = learn.data.classes[np.argmax(losses)]
    template = 'prediction.html'
    context = {'request': request, 'prediction':pred,
            'pigeon':losses[0].item(), 'turkey':losses[1].item()}
    return templates.TemplateResponse(template, context)

templates = Jinja2Templates(directory='templates')

path = Path('./model')
learn = load_learner(path)

app = Starlette()

@app.route("/")
def form(request):
    template = "home.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return get_prediction(bytes, request)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return get_prediction(bytes, request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
