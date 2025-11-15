import argparse, os, cv2, numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
import torch.nn as nn
import builtins, importlib, pkgutil

# ---- allowlist all trusted globals ----
safe = {
    tasks.SegmentationModel,
    modules.conv.Conv,
    modules.block.C2f,
    modules.head.Detect,
    builtins.getattr, builtins.setattr, builtins.len, builtins.range
}

# add every torch.nn layer class
for _, mname, _ in pkgutil.walk_packages(nn.__path__, nn.__name__ + "."):
    try:
        mod = importlib.import_module(mname)
        for _, obj in vars(mod).items():
            if isinstance(obj, type):
                safe.add(obj)
    except Exception:
        pass

# register them once
torch.serialization.add_safe_globals(list(safe))
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"Loading FastSAM (YOLOv8-seg style) model: {args.model}")
model = YOLO(args.model, task="segment")  # now loads safely, any internal class allowed

print("Running segmentation inference...")
results = model.predict(args.image, imgsz=640, device="cpu", verbose=False)

orig = Image.open(args.image)
orig_w, orig_h = orig.size
mask_dir = args.output_dir
count = 0

for r in results:
    if not hasattr(r, "masks") or r.masks is None:
        print("⚠️ No masks detected in this image.")
        continue

    masks = r.masks.data.cpu().numpy()
    for i, m in enumerate(masks):
        # m: (H, W) float mask from YOLO output
        mask = m

        # ---- remove letterbox padding before resize ----
        r = min(640 / orig_h, 640 / orig_w)
        new_w, new_h = round(orig_w * r), round(orig_h * r)
        pad_w, pad_h = (640 - new_w) / 2, (640 - new_h) / 2
        x0, x1 = int(pad_w), int(pad_w + new_w)
        y0, y1 = int(pad_h), int(pad_h + new_h)

        # clip to valid range (defensive)
        y0, y1 = max(0, y0), min(m.shape[0], y1)
        x0, x1 = max(0, x0), min(m.shape[1], x1)

        cropped = m[y0:y1, x0:x1]
        # resize to original image
        mask = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(mask_dir, f"mask_{count}.png"), mask)
        count += 1


print(f"✅ Saved {count} mask(s) to {mask_dir}")
