import os
import base64
import sys
import json
import googleapiclient.discovery
import io
import httplib2
import urllib

from PIL import Image
from urllib import request, parse
from urllib.request import urlopen
from flask import Flask, render_template, request
from google.cloud import logging

app = Flask(__name__)
client = logging.Client()
logger = client.logger("mlhotdog")

@app.route("/", methods=["GET"])
def home():
  return render_template("index.html")

@app.route("/tasks/keepmlalive")
def keep_ml_alive():
  logger.log_struct({
      "task": "keepmlalive",
      "urlRoot": request.url_root,
  })

  data = json.dumps({"fileUrl": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Hot_dog_with_mustard.png/1200px-Hot_dog_with_mustard.png"}).encode()
  req = urllib.request.Request(request.url_root + "api/predict", data=data, headers={"content-type": "application/json"})
  resp = urllib.request.urlopen(req)
  data = resp.read()
  return "DONE"

@app.route("/api/predict", methods=["POST"])
def predict():
  print("Start of request")
  if request.data is None or not request.data.decode("utf-8"):
    logger.log_struct({
        "error": "no request body",
        "request": request}, severity="ERROR")
    return json.dumps({"error": "no request body"}), 400

  print("Parsing payload as json")
  try:
    payload = json.loads(request.data.decode("utf-8"))
  except:
    logger.log_struct({
        "error": "invalid json post payload",
        "exception": sys.exc_info()}, severity="ERROR")
    return json.dumps({"error": "invalid json post payload"}), 400

  print("Getting image")
  img = None
  if "fileUrl" in payload:
    try:
      img = base64.b64encode(urlopen(payload.get("fileUrl")).read())
    except:
      logger.log_text({"error": "failed to fetch image", "url": payload.get("fileUrl"), "exception": sys.exc_info()})
      return json.dumps({"error": "failed to fetch image", "url": payload.get("fileUrl")}), 500
  elif "fileBinary" in payload:
    img = payload.get("fileBinary", None).split(",")[1]
  else:
    logger.log_struct({"error": "no image"}, severity="ERROR")
    return json.dumps({"error": "no image"}), 400

  # Read in image as an Image in memory
  print("Reading image in memory")
  try:
    buff = Image.open(io.BytesIO(base64.b64decode(img)))
  except:
    logger.log_struct({"error": "not given valid image", "exception": sys.exc_info()}, severity="ERROR")
    return json.dumps({"error": "Not given a valid image"}), 400

  # Resize image to be at max 1024x1024
  print("Resizing image")
  try:
    size = 1024, 1024
    buff.thumbnail(size, Image.ANTIALIAS)

    # Convert image to JPEG
    out_buff = io.BytesIO()
    buff.convert("RGB").save(out_buff, format="JPEG")

    # Output image as base64
    out_buff.seek(0)
    img = base64.b64encode((out_buff.getvalue())).decode("ascii")

  except:
    logger.log_struct({"error": "Could not manipulate image to be a jpeg of 1024x1024", "exception": sys.exc_info()}, severity="ERROR")
    return json.dumps({"error": "Could not manipulate image to be a jpeg of 1024x1024"}), 500

  print("Starting to call ml service")
  try:
    service = googleapiclient.discovery.build("ml", "v1")
    resp = service.projects().predict(
        name="projects/jcham-1469824226729/models/hotdog",
        body={
            "instances": [{
                "key": "0",
                "image_bytes": {
                    "b64": img,
                },
            }]
        }).execute()
    logger.log_struct({"predictionResp": resp}, severity="DEBUG")
  except:
    logger.log_struct({"error": "Failed to get ML Prediction", "exception": sys.exc_info()}, severity="ERROR")
    return json.dumps({"error": "Failed to get ML prediction"}), 500

  return json.dumps(resp)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8082)
