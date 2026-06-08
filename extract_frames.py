from pathlib import Path
import cv2

training_dir = Path("training")
output_dir = Path("frames")

output_dir.mkdir(exist_ok=True)

for video_path in sorted(training_dir.glob("*.mp4")):
    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_id = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_name = video_output_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_name), frame)

        frame_id += 1

    cap.release()

    print(f"{video_name}: extracted {frame_id} frames")