import os
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from moviepy.editor import VideoFileClip, ImageSequenceClip
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForImageSegmentation

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

# Load BiRefNet models
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device)
birefnet_lite = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True).to(device)

# Transformations
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Fixed background image path (hardcoded)
BACKGROUND_IMAGE_PATH = r"E:\mywork\Bhalu\video-background-removal\blvk.webp"

def process_frame(frame, fast_mode):
    try:
        pil_image = Image.fromarray(frame)
        
        # Use the fixed background image
        bg_image = Image.open(BACKGROUND_IMAGE_PATH).convert("RGBA")
        
        # Debug: Check if the background image is correctly loaded
        print(f"Background image size: {bg_image.size}")
        
        processed_image = process(pil_image, bg_image, fast_mode)
        
        return np.array(processed_image)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame


def process(image, bg, fast_mode):
    global device
    try:
        input_images = transform_image(image).unsqueeze(0).to(device)
        model = birefnet_lite if fast_mode else birefnet

        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)

        mask.save("temp_files/mask.png")
        
        background = bg.convert("RGBA").resize(image.size)

        return Image.composite(image, background, mask)
    except RuntimeError as e:
        if "CUDA" in str(e):
            device = "cpu"
            input_images = transform_image(image).unsqueeze(0)
            model = model.to(device)
            with torch.no_grad():
                preds = model(input_images)[-1].sigmoid()
            pred = preds[0].squeeze()
            return transforms.ToPILImage()(pred)
        else:
            raise e


def process_video(input_video_path, output_video_path, fps=0, fast_mode=True, max_workers=6):
    try:
        print(f"Input video path: {input_video_path}")
        print(f"Background image path: {BACKGROUND_IMAGE_PATH}")
        print(f"Output video path: {output_video_path}")  # Add this line to check the output path

        start_time = time.time()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_video_path)
        print(f"Output directory: {output_dir}")  # Add this line to check the output directory

        if not os.path.exists(output_dir) and output_dir:
            os.makedirs(output_dir)

        # Load the input video
        video = VideoFileClip(input_video_path)
        fps = fps or video.fps
        audio = video.audio
        frames = list(video.iter_frames(fps=fps))

        # Process each frame in parallel
        processed_frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_frame, frames[i], fast_mode) for i in range(len(frames))
            ]

            # Collect processed frames and print progress
            for i, future in enumerate(futures):
                result = future.result()
                processed_frames.append(result)
                print(f"Processing: {((i + 1) / len(frames)) * 100:.2f}%")

        # Create the final video from processed frames
        processed_video = ImageSequenceClip(processed_frames, fps=fps).set_audio(audio)

        # Save the output video to the correct path
        print(f"Saving video to: {output_video_path}")
        processed_video.write_videofile(output_video_path, codec="libx264")

        # Calculate and print the processing time
        elapsed_time = time.time() - start_time
        print(f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error processing video: {e}")


# Example usage
if __name__ == "__main__":
    input_video = r"E:\mywork\Bhalu\video-background-removal\input_video.mp4"  # Path to the input video
    output_video = "output.mp4"  # Path to save the output video
    fps = 0  # FPS of the output video, 0 to inherit the original FPS
    fast_mode = True  # Use BiRefNet_lite if True
    max_workers = 6  # Number of threads for parallel frame processing

    process_video(input_video, output_video, fps=fps, fast_mode=fast_mode, max_workers=max_workers)
