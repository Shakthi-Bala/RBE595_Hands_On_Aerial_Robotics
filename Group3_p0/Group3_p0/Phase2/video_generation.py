import cv2
import glob
import os
import numpy as np

render_dir = "/home/alien/MyDirectoryID_p0/Phase2/render"     
out_path   = "rgb_depth.mp4"

rgb_files   = sorted(glob.glob(os.path.join(render_dir, "rgb_*.jpg")))
depth_files = sorted(glob.glob(os.path.join(render_dir, "depth_*.jpg")))


assert len(rgb_files) == len(depth_files), "RGB/Depth count mismatch"


rgb0   = cv2.imread(rgb_files[0])
depth0 = cv2.imread(depth_files[0])


if depth0.shape != rgb0.shape:
    depth0 = cv2.resize(depth0, (rgb0.shape[1], rgb0.shape[0]))

concat0 = np.hstack((rgb0, depth0))

h, w, _ = concat0.shape
fps = 20  

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for rf, df in zip(rgb_files, depth_files):
    rgb   = cv2.imread(rf)
    depth = cv2.imread(df)
    if depth.shape != rgb.shape:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
    frame = np.hstack((rgb, depth))
    writer.write(frame)

writer.release()
print(f"Saved video to {out_path}")
