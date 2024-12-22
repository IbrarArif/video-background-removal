from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from uuid import uuid4
from app import process_video  # Import your existing code for video processing
from PIL import Image
import io

app = FastAPI()

# Enable CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Temporary directory for saving uploaded files and output
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/remove_background/")
async def remove_background(
    input_video: UploadFile,
):
    input_video_path = os.path.join(TEMP_DIR, f"{uuid4()}.mp4")
    output_video_path = os.path.join(TEMP_DIR, f"{uuid4()}_output.mp4")

    try:
        # Save uploaded input video
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(input_video.file, f)

        # Call the process_video function with fixed parameters
        process_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            fps=0,  # Using the original video FPS
            fast_mode=True,  # Default setting for fast_mode
            max_workers=6  # Default setting for max workers
        )

        # Return the processed video file
        response = FileResponse(output_video_path, media_type="video/mp4")

        return response

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to check if the API is working
@app.get("/")
async def root():
    return {"message": "Video Background Remover API is up and running!"}
