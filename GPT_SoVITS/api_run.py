import uvicorn

if __name__ == "__main__":
    uvicorn.run("inference_stream_api:app", host="127.0.0.1", port=5000, workers=1, reload=False)