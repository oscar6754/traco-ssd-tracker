from pathlib import Path
import pandas as pd
import cv2

training_dir = Path("training")
frames_dir = Path("frames")
output_csv = Path("ssd_annotations.csv")
check_dir = Path("label_check")

check_dir.mkdir(exist_ok=True)

BOX_RADIUS = 30
SAVE_CHECK_IMAGES = False
SAMPLE_EVERY = 10

rows = []
skipped_boxes = 0

for csv_path in sorted(training_dir.glob("*.csv")): # use all csv
    video_name = csv_path.stem
    frame_folder = frames_dir / video_name

    if not frame_folder.exists():
        print(f"Frame folder not avaliable: {frame_folder}")
        continue


    # for reading files
    df = pd.read_csv(csv_path)
    
    

    for t, group in df.groupby("t"):
        t = int(t)
        
        # 06d 6 digts zero in front
        frame_path = frame_folder / f"frame_{t:06d}.jpg"

        if not frame_path.exists():
            print(f"Missing frame: {frame_path}")
            continue


        # load image y get size
        image = cv2.imread(str(frame_path))
        height, width = image.shape[:2]

        for _, row in group.iterrows():
            x = float(row["x"])
            y = float(row["y"])

            xmin = min(width - 1, max(0, x - BOX_RADIUS))
            ymin = min(height - 1, max(0, y - BOX_RADIUS))
            xmax = min(width - 1, max(0, x + BOX_RADIUS))
            ymax = min(height - 1, max(0, y + BOX_RADIUS))

            if xmax <= xmin or ymax <= ymin:
                skipped_boxes += 1
                continue

            rows.append({
                "image_path": str(frame_path),
                "video": video_name,
                "t": t,
                "hexbug": int(row["hexbug"]),
                "x": x,
                "y": y,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": 1
            })


        # just for checking, could skip
        if SAVE_CHECK_IMAGES and t % SAMPLE_EVERY == 0:
            check_video_dir = check_dir / video_name
            check_video_dir.mkdir(exist_ok=True)

            for _, row in group.iterrows():
                x = int(row["x"])
                y = int(row["y"])
                hexbug_id = int(row["hexbug"])

                
                
                
                xmin = min(width - 1, max(0, x - BOX_RADIUS))
                ymin = min(height - 1, max(0, y - BOX_RADIUS))
                xmax = min(width - 1, max(0, x + BOX_RADIUS))
                ymax = min(height - 1, max(0, y + BOX_RADIUS))

                if xmax <= xmin or ymax <= ymin:
                    continue

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(
                    image,
                    str(hexbug_id),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            out_path = check_video_dir / f"frame_{t:06d}.jpg"
            cv2.imwrite(str(out_path), image)

    print(f"Processed {video_name}")

annotations = pd.DataFrame(rows)
annotations.to_csv(output_csv, index=False)

print(f"\nSaved: {output_csv}")
print(f"Total boxes: {len(annotations)}")
print(f"Total images with labels: {annotations['image_path'].nunique()}")
print(f"Skipped invalid boxes: {skipped_boxes}")
