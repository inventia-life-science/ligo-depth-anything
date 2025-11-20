import glob
import os

import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export import export_to_depth_vis


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the smaller model for 4 GB VRAM
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
    model = model.to(device=device)

    # Example images from the repo
    example_path = "assets/examples/SOH"
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))

    # Run inference; lower process_res to save VRAM
    prediction = model.inference(
        images,
        process_res=384,
        process_res_method="upper_bound_resize",
    )

    print("processed_images:", prediction.processed_images.shape)
    print("depth:", prediction.depth.shape)
    print("conf:", prediction.conf.shape)

    export_dir = "outputs"
    os.makedirs(export_dir, exist_ok=True)
    export_to_depth_vis(prediction, export_dir)

if __name__ == "__main__":
    main()