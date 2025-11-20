import os

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export import export_to_depth_vis


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model choice: BASE may OOM on 4 GB; switch to DA3-SMALL if needed.
    model = DepthAnything3.from_pretrained(
    "/home/tom/repos/Depth-Anything-3/models/DA3-BASE"
    )
    model = model.to(device=device)

    # Capture a single frame from the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        raise RuntimeError("Failed to capture frame from webcam.")

    # Convert BGR (OpenCV) to RGB, as the model expects RGB images
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Run inference on a single image
    prediction = model.inference(
        [frame_rgb],
        process_res=800,
        process_res_method="upper_bound_resize",
    )

    print("processed_images:", prediction.processed_images.shape)
    print("depth:", prediction.depth.shape)
    print("conf:", prediction.conf.shape)

    # Extract estimated camera poses
    if prediction.extrinsics is not None:
        print("\n=== Estimated Camera Extrinsics (4x4) ===")
        print(prediction.extrinsics[0])  # First (and only) camera
        
        # Extract camera position (translation from extrinsics)
        # Extrinsics is world-to-camera, so camera position in world is -R^T * t
        R = prediction.extrinsics[0][:3, :3]
        t = prediction.extrinsics[0][:3, 3]
        camera_pos = -R.T @ t
        print(f"\nCamera position (world space): {camera_pos}")
        print(f"Camera translation (w2c): {t}")
        
    if prediction.intrinsics is not None:
        print("\n=== Estimated Camera Intrinsics (3x3) ===")
        print(prediction.intrinsics[0])  # First (and only) camera
        K = prediction.intrinsics[0]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}")
        print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

    export_dir = "outputs_webcam"
    os.makedirs(export_dir, exist_ok=True)
    export_to_depth_vis(prediction, export_dir)
    print(f"\nSaved depth visualization to: {os.path.join(export_dir, 'depth_vis')}")
    
    # Save poses to file
    if prediction.extrinsics is not None and prediction.intrinsics is not None:
        pose_file = os.path.join(export_dir, "camera_poses.npz")
        np.savez(
            pose_file,
            extrinsics=prediction.extrinsics,
            intrinsics=prediction.intrinsics,
        )
        print(f"Saved camera poses to: {pose_file}")


if __name__ == "__main__":
    main()



