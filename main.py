import argparse
import os
import numpy as np
import open3d as o3d
from PIL import Image, ImageFilter, ImageEnhance
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_PATH   = r"./sample.obj"
DEFAULT_TARGET = "pillow"
NUM_POINTS     = 200_000
IMG_SIZE       = 640
DEBUG_DIR      = "debug_views"

# These are only used if --no_auto_profile is passed
MIN_VOTES          = 2
MIN_BOX_AREA_FRAC  = 0.01
MAX_BOX_AREA_FRAC  = 0.50
DEPTH_ZSCORE_THRESH = 1.5

# Auto-profile: scout views used to measure object size before main loop
SCOUT_VIEWS = [
    ("top",   0, 1, False, False),
    ("front", 0, 2, False, False),
    ("left",  1, 2, False, False),
]

VIEWS = [
    ("front",      0, 2, False, False),
    ("back",       0, 2, True,  False),
    ("left",       1, 2, False, False),
    ("right",      1, 2, True,  False),
    ("top",        0, 1, False, False),
    ("bottom",     0, 1, False, True ),
    ("diag_front", 0, 2, False, False),
    ("diag_side",  1, 2, False, False)
]

# ── Device ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device = {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. FLORENCE-2-LARGE-FT
# ═════════════════════════════════════════════════════════════════════════════
print("[init] Loading Florence-2-large-ft …")
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
).to(device).eval()
print("[init] Florence-2-large-ft ready.")

# ═════════════════════════════════════════════════════════════════════════════
# 2. DEPTH ANYTHING V2
# ═════════════════════════════════════════════════════════════════════════════
print("[init] Loading DepthAnything-v2-small …")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=device,
)
print("[init] DepthAnything v2 ready.\n")


# ═════════════════════════════════════════════════════════════════════════════
# 3. FLORENCE GROUNDING
# ═════════════════════════════════════════════════════════════════════════════
def run_florence_ovd(image: Image.Image, target_text: str):
    """OPEN_VOCABULARY_DETECTION with label filtering."""
    prompt = f"<OPEN_VOCABULARY_DETECTION> {target_text}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        result,
        task="<OPEN_VOCABULARY_DETECTION>",
        image_size=(image.width, image.height),
    )

    data   = parsed.get("<OPEN_VOCABULARY_DETECTION>", {})
    bboxes = data.get("bboxes", [])
    labels = data.get("bboxes_labels", [])

    return [
        bbox for bbox, label in zip(bboxes, labels)
        if target_text.lower() in label.lower()
    ]


def run_florence_dense(image: Image.Image, target_text: str):
    """
    DENSE_REGION_CAPTION fallback — finds many small regions.
    Better for thin/small objects that OVD misses.
    """
    prompt  = "<DENSE_REGION_CAPTION>"
    inputs  = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        result,
        task="<DENSE_REGION_CAPTION>",
        image_size=(image.width, image.height),
    )

    data   = parsed.get("<DENSE_REGION_CAPTION>", {})
    bboxes = data.get("bboxes", [])
    labels = data.get("labels", [])

    return [
        bbox for bbox, label in zip(bboxes, labels)
        if target_text.lower() in label.lower()
    ]


def run_florence_grounding(image: Image.Image, target_text: str):
    """Try OVD first; fall back to DENSE_REGION_CAPTION if nothing found."""
    bboxes = run_florence_ovd(image, target_text)

    if not bboxes:
        print("   [fallback] OVD → 0 detections, trying DENSE_REGION_CAPTION …")
        bboxes = run_florence_dense(image, target_text)
        if bboxes:
            print(f"   [fallback] DENSE found {len(bboxes)} region(s)")

    return bboxes


# ═════════════════════════════════════════════════════════════════════════════
# 4. MESH LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_mesh(file_path: str):
    print(f"[load] {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=True)
    mesh.compute_vertex_normals()

    if mesh.has_vertex_colors():
        print("[load] vertex colors present ✓")
        return mesh

    print("[load] No vertex colors — trying texture bake …")
    try:
        import trimesh
        tm = trimesh.load(file_path, force="mesh")
        if hasattr(tm.visual, "to_color"):
            vc = tm.visual.to_color().vertex_colors[:, :3] / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
            print("[load] texture baked ✓")
    except Exception as e:
        print(f"[load] texture bake failed ({e}), will use depth colors")

    return mesh


# ═════════════════════════════════════════════════════════════════════════════
# 5. DEPTH-BASED POINT COLORS
# ═════════════════════════════════════════════════════════════════════════════
def compute_depth_colors(pcd, depth_axis: int = 2):
    points = np.asarray(pcd.points)
    d      = points[:, depth_axis]
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    cmap   = plt.get_cmap("viridis")
    return cmap(d_norm)[:, :3]


# ═════════════════════════════════════════════════════════════════════════════
# 6. PROJECTION
# ═════════════════════════════════════════════════════════════════════════════
def project(points, colors, ax_h, ax_v, flip_h=False, flip_v=False):
    h        = points[:, ax_h].copy()
    v        = points[:, ax_v].copy()
    depth_ax = 3 - ax_h - ax_v
    d        = points[:, depth_ax]

    if flip_h: h = h.max() - h
    if flip_v: v = v.max() - v

    def norm(arr, size):
        return ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * (size - 1)).astype(int)

    px_col = norm(h, IMG_SIZE)
    px_row = (IMG_SIZE - 1) - norm(v, IMG_SIZE)

    order   = np.argsort(-d)
    rgb     = (colors[order] * 255).astype(np.uint8)

    canvas  = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    idx_map = np.full((IMG_SIZE, IMG_SIZE), -1, dtype=np.int32)

    canvas [px_row[order], px_col[order]] = rgb
    idx_map[px_row[order], px_col[order]] = order

    return Image.fromarray(canvas), idx_map


def postprocess_image(img):
    img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# 7. BOX AREA FILTER
# ═════════════════════════════════════════════════════════════════════════════
def filter_boxes_by_area(bboxes, img_w, img_h, min_frac, max_frac):
    total    = img_w * img_h
    filtered = []
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        frac = max(0, x2 - x1) * max(0, y2 - y1) / total
        if min_frac <= frac <= max_frac:
            filtered.append(box)
        else:
            print(f"   [area-filter] dropped frac={frac:.4f} "
                  f"(limits [{min_frac:.4f}, {max_frac:.2f}])")
    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# 8. AUTO-PROFILE 
# ═════════════════════════════════════════════════════════════════════════════
def auto_profile(pcd, colors, target: str) -> dict:
    print("\n[auto-profile] Starting scout pass …")
    points = np.asarray(pcd.points)
    all_fracs = []
    views_with_detection = 0  # ← track how many views found something

    for (label, ax_h, ax_v, flip_h, flip_v) in SCOUT_VIEWS:
        depth_axis  = 3 - ax_h - ax_v
        view_colors = compute_depth_colors(pcd, depth_axis=depth_axis) \
                      if not pcd.has_colors() else colors

        img, _ = project(points, view_colors, ax_h, ax_v, flip_h, flip_v)
        img     = postprocess_image(img)
        W, H    = img.size

        bboxes = run_florence_grounding(img, target)
        bboxes = filter_boxes_by_area(bboxes, W, H,
                                      min_frac=0.001, max_frac=0.95)

        view_had_detection = False
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            frac = max(0, x2 - x1) * max(0, y2 - y1) / (W * H)
            all_fracs.append(frac)
            view_had_detection = True
            print(f"   [{label}] detected box frac={frac:.4f}")

        if view_had_detection:
            views_with_detection += 1

    print(f"[auto-profile] scout views with valid detection: "
          f"{views_with_detection}/{len(SCOUT_VIEWS)}")

    if not all_fracs:
        profile = {
            "min_votes":    1,
            "min_area":     0.001,
            "max_area":     0.30,
            "depth_zscore": 2.5,
            "size_class":   "small (fallback)",
        }
    else:
        median_frac = float(np.median(all_fracs))
        print(f"[auto-profile] median box area fraction = {median_frac:.4f}")

        # ── Size class  ──────────────────────────────────
        if median_frac > 0.10:
            size_class   = "large"
            min_area     = round(median_frac * 0.30, 4)
            max_area     = 0.80
            depth_zscore = 1.5
        elif median_frac > 0.02:
            size_class   = "medium"
            min_area     = round(median_frac * 0.20, 4)
            max_area     = 0.55
            depth_zscore = 2.0
        else:
            size_class   = "small"
            min_area     = round(median_frac * 0.10, 4)
            max_area     = 0.30
            depth_zscore = 2.5

        # ── min_votes: driven by detection coverage ──────────────────────
        # If the object was only seen in 1 scout view, it's likely
        # geometrically sparse — don't demand cross-view confirmation
        if views_with_detection <= 1:
            min_votes  = 1
            size_class += " (sparse)"
        elif views_with_detection == 2:
            min_votes  = 2
        else:
            min_votes  = 3  # seen in all 3 scout views → be strict

        profile = {
            "min_votes":    min_votes,
            "min_area":     min_area,
            "max_area":     max_area,
            "depth_zscore": depth_zscore,
            "size_class":   size_class,
        }

    print(f"[auto-profile] → size_class = {profile['size_class']}")
    print(f"[auto-profile] → min_votes={profile['min_votes']}, "
          f"min_area={profile['min_area']}, max_area={profile['max_area']}, "
          f"depth_zscore={profile['depth_zscore']}\n")
    return profile

# ═════════════════════════════════════════════════════════════════════════════
# 9. DEPTH LIFTING
# ═════════════════════════════════════════════════════════════════════════════
def depth_lift_mask(img: Image.Image, box, zscore_thresh: float):
    x1, y1, x2, y2 = map(int, box)
    W, H = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    if x2 <= x1 or y2 <= y1:
        mask = np.zeros((H, W), dtype=bool)
        mask[y1:y2+1, x1:x2+1] = True
        return mask

    depth_result = depth_pipe(img)
    depth_np     = np.array(depth_result["depth"])
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_norm   = (depth_np - d_min) / (d_max - d_min + 1e-8)

    roi_depth = depth_norm[y1:y2+1, x1:x2+1]
    median    = np.median(roi_depth)
    std       = roi_depth.std()

    mask = np.zeros((H, W), dtype=bool)

    if std < 1e-6:
        mask[y1:y2+1, x1:x2+1] = True
        return mask

    z_scores = np.abs(roi_depth - median) / std
    roi_mask = z_scores <= zscore_thresh
    mask[y1:y2+1, x1:x2+1] = roi_mask

    kept_frac = roi_mask.sum() / roi_mask.size
    print(f"   [depth-lift] kept {kept_frac*100:.1f}% of bbox pixels "
          f"(median={median:.3f}, std={std:.3f}, thresh={zscore_thresh})")
    return mask


# ═════════════════════════════════════════════════════════════════════════════
# 10. BACKPROJECTION
# ═════════════════════════════════════════════════════════════════════════════
def backproject(bboxes, idx_map, img, use_depth_lift: bool, zscore_thresh: float):
    indices = set()
    for box in bboxes:
        if use_depth_lift:
            mask = depth_lift_mask(img, box, zscore_thresh)
            pts  = idx_map[mask]
            indices.update(pts[pts >= 0])
        else:
            x1, y1, x2, y2 = map(int, box)
            region = idx_map[y1:y2+1, x1:x2+1]
            indices.update(region[region >= 0])
    return indices


# ═════════════════════════════════════════════════════════════════════════════
# 11. MULTI-VIEW VOTING
# ═════════════════════════════════════════════════════════════════════════════
def multi_view_vote(pcd, colors, target, profile: dict, use_depth_lift: bool):
    points      = np.asarray(pcd.points)
    vote_counts = np.zeros(len(points), dtype=np.int32)
    min_votes   = profile["min_votes"]
    min_area    = profile["min_area"]
    max_area    = profile["max_area"]
    zscore      = profile["depth_zscore"]

    os.makedirs(DEBUG_DIR, exist_ok=True)

    for (label, ax_h, ax_v, flip_h, flip_v) in VIEWS:
        print(f"\n[{label}] projecting …")

        depth_axis  = 3 - ax_h - ax_v
        view_colors = compute_depth_colors(pcd, depth_axis=depth_axis) \
                      if not pcd.has_colors() else colors

        img, idx_map = project(points, view_colors, ax_h, ax_v, flip_h, flip_v)
        img          = postprocess_image(img)
        img.save(os.path.join(DEBUG_DIR, f"{label}.png"))

        bboxes = run_florence_grounding(img, target)
        print(f"  raw detections   : {len(bboxes)}")

        bboxes = filter_boxes_by_area(bboxes, img.width, img.height, min_area, max_area)
        print(f"  after area-filter: {len(bboxes)}")

        if not bboxes:
            continue

        matched = backproject(bboxes, idx_map, img, use_depth_lift, zscore)
        vote_counts[list(matched)] += 1
        print(f"  3D points voted  : {len(matched)}")

    final_indices = set(np.where(vote_counts >= min_votes)[0])
    print(f"\n[vote] Points surviving >= {min_votes} votes: {len(final_indices)}")
    return final_indices


# ═════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═════════════════════════════════════════════════════════════════════════════
def segment_3d(file_path, target, no_auto_profile,
               min_votes, min_area, max_area, depth_zscore, no_depth_lift):

    mesh = load_mesh(file_path)
    pcd  = mesh.sample_points_poisson_disk(NUM_POINTS)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        print("[color] using baked vertex colors")
    else:
        colors = compute_depth_colors(pcd, depth_axis=2)
        print("[color] using depth-based viridis colors")

    # ── Profile resolution ───────────────────────────────────────────────────
    if no_auto_profile:
        # Use whatever was passed via CLI (or script defaults)
        profile = {
            "min_votes":    min_votes,
            "min_area":     min_area,
            "max_area":     max_area,
            "depth_zscore": depth_zscore,
            "size_class":   "manual",
        }
        print(f"[config] manual profile: {profile}")
    else:
        profile = auto_profile(pcd, colors, target)

        # CLI overrides take precedence over auto-profile when explicitly set
        if min_votes   != MIN_VOTES:           profile["min_votes"]    = min_votes
        if min_area    != MIN_BOX_AREA_FRAC:   profile["min_area"]     = min_area
        if max_area    != MAX_BOX_AREA_FRAC:   profile["max_area"]     = max_area
        if depth_zscore != DEPTH_ZSCORE_THRESH: profile["depth_zscore"] = depth_zscore

    use_depth_lift = not no_depth_lift
    print(f"[config] depth_lift={use_depth_lift}\n")

    final_indices = multi_view_vote(pcd, colors, target, profile, use_depth_lift)

    if not final_indices:
        print("[result] No points survived.")
        print("  → Try --no_auto_profile with --min_votes 1 --min_area 0.001 --depth_zscore 2.5")
        return

    vis = np.full((len(pcd.points), 3), 0.45)
    vis[list(final_indices)] = [1.0, 0.08, 0.08]
    pcd.colors = o3d.utility.Vector3dVector(vis)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Segmented: {target}",
        width=1280,
        height=720,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D segmentation: Florence-2 + auto-profile + depth lifting + voting"
    )
    parser.add_argument("--path",             default=DEFAULT_PATH)
    parser.add_argument("--target",           default=DEFAULT_TARGET,
                        help="Object to segment e.g. 'pillows', 'lamp', 'leg'")

    # Auto-profile
    parser.add_argument("--no_auto_profile",  action="store_true",
                        help="Skip auto-profile and use manual threshold args")

    # Manual threshold overrides (also override auto-profile when passed)
    parser.add_argument("--min_votes",        type=int,   default=MIN_VOTES)
    parser.add_argument("--min_area",         type=float, default=MIN_BOX_AREA_FRAC)
    parser.add_argument("--max_area",         type=float, default=MAX_BOX_AREA_FRAC)
    parser.add_argument("--depth_zscore",     type=float, default=DEPTH_ZSCORE_THRESH)
    parser.add_argument("--no_depth_lift",    action="store_true")

    args, _ = parser.parse_known_args()

    segment_3d(
        file_path       = args.path,
        target          = args.target,
        no_auto_profile = args.no_auto_profile,
        min_votes       = args.min_votes,
        min_area        = args.min_area,
        max_area        = args.max_area,
        depth_zscore    = args.depth_zscore,
        no_depth_lift   = args.no_depth_lift,
    )
