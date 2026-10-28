import cv2
import numpy as np
import math
import torch
import sys
import os
import csv
from collections import deque, defaultdict
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor
from sklearn.linear_model import LinearRegression
import joblib
import argparse

# =========================================================
# 0. COMMAND-LINE ARGUMENTS (no hardcoded machine paths)
# =========================================================
parser = argparse.ArgumentParser(
    description="Road Safety Audit: lane, curvature, width and sign-clearance telemetry from dashcam video."
)
parser.add_argument("--video", required=True, help="Path to the input road video")
parser.add_argument("--yolo-weights", required=True, help="Path to the trained YOLO segmentation weights (best.pt)")
parser.add_argument("--depth-checkpoint", required=True,
                    help="Path to depth_anything_v2_metric_vkitti_vitb.pth")
parser.add_argument("--depth-anything-dir", default="Depth-Anything-V2",
                    help="Path to the cloned Depth-Anything-V2 repository (default: ./Depth-Anything-V2)")
parser.add_argument("--sam3-weights", default="sam3.pt",
                    help="Path to sam3.pt (only needed for MPP_Y option 2 'SAM 3')")
parser.add_argument("--output-dir", default="outputs", help="Directory for output videos and CSV log")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
# =========================================================
# 1. CAMERA PARAMETERS & EXACT TARGET CALCULATION
# =========================================================
print("[INFO] Calculating exact Z_optical targets...")

# ---> UPDATE THIS IF YOUR CAMERA IS NOT 1.5 METERS OFF THE GROUND <---
H_c = 1.25           # Camera height in meters
pitch_deg = 8.0     # Camera downward pitch in degrees

# Convert to radians for math functions
theta = np.radians(pitch_deg)

# The physical ground marks where you placed the cones
z_ground_points = np.array([
    10, 10, 10,  # 10m marks (left, center, right)
    20, 20, 20,  # 20m marks (left, center, right)
    30, 30, 30   # 30m marks (left, center, right)
])

# Mathematically convert ground distance to perfect optical distance
y_true_distances = (z_ground_points * np.cos(theta)) + (H_c * np.sin(theta))

for g, opt in zip([10, 20, 30], [y_true_distances[0], y_true_distances[3], y_true_distances[6]]):
    print(f"   Ground: {g}m  ->  Z_optical: {opt:.3f}m")


# =========================================================
# 2. PREPARE YOUR DATASET
# =========================================================
# X: Input features (The raw AI Distances printed on your cones)
X_ai_distances = np.array([
    [24.74], [29.99], [31.36],   # 10-meter frame
    [55.35], [55.94], [58.38],   # 20-meter frame
    [72.93], [75.75], [74.03]    # 30-meter frame 
])


# =========================================================
# 3. TRAIN THE LINEAR REGRESSION MODEL
# =========================================================
print("\n[INFO] Initializing Linear Regression Model...")
model = LinearRegression()

# Train the model mapping the AI's guesses to the true optical distances
model.fit(X_ai_distances, y_true_distances)


# =========================================================
# 4. EXTRACT AND INSPECT THE CALIBRATION PARAMETERS
# =========================================================
scale = model.coef_[0]      # The slope (s)
shift = model.intercept_  # The y-intercept (t)

print("\n=== CALIBRATION RESULTS ===")
print(f"Calculated Scale (s): {scale:.4f}")
print(f"Calculated Shift (t): {shift:.4f}")
print(f"Model Quality (R² Score): {model.score(X_ai_distances, y_true_distances):.4f}") 


# =========================================================
# 5. SAVE THE MODEL FOR YOUR CAR
# =========================================================
model_filename = "distance_calibration_model.pkl"
joblib.dump(model, model_filename)
print(f"\n[SUCCESS] Calibration model saved as '{model_filename}'")
# Ensure Depth Anything V2 pathing works
sys.path.append(os.path.abspath(args.depth_anything_dir))
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# =========================================================================
# CONFIGURATIONS
# =========================================================================
SAM3_MODEL_PATH = args.sam3_weights
 

# =========================================================================
# 1. LOAD VIDEO FIRST 
# =========================================================================
input_video = args.video
_stem = os.path.splitext(os.path.basename(input_video))[0]
output_video = os.path.join(args.output_dir, f"{_stem}_audit.mp4")
output_bev_video = os.path.join(args.output_dir, f"{_stem}_bev.mp4")

cap = cv2.VideoCapture(input_video)
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if v_width == 0 or v_height == 0:
    print("[ERROR] Could not load video. Check the file path.")
    sys.exit()

# =========================================================================
# 2. DYNAMIC SETUP & HARDCODED CALIBRATION
# =========================================================================
print("\n" + "="*50)
print("ROAD SAFETY AUDIT TELEMETRY SYSTEM - INITIALIZING")
print("="*50)
print(f"[INFO] Video Detected: {v_width}x{v_height} @ {fps} FPS")

EXPECTED_LANES = int(input("Enter the expected number of lanes on this road (e.g., 2, 4, 6): "))
EXPECTED_CENTERS = max(0, EXPECTED_LANES - 1)
EXPECTED_SHOULDERS = 2
camera_height_hc = float(input("Enter camera mounting height in meters (Press Enter for 1.25): ") or "1.25")
radius_threshold = float(input("Enter minimum safe Radius threshold (in meters): "))
width_threshold = float(input("Enter minimum safe Lane Width threshold (in meters): "))
HAS_CRASH_BARRIER = input("Is there a crash barrier in the video? (yes/no): ").strip().lower() in ['yes', 'y']
HAS_KERB = input("Is there a kerb in the video? (yes/no): ").strip().lower() in ['yes', 'y']
GENERATE_BEV = input("Do you want to generate and display the BEV video? (yes/no): ").strip().lower() in ['yes', 'y']
synchronized_timestamp = float(input("Enter starting timestamp (in seconds) to run calibration: "))
CALIBRATION_SYNC_TS = synchronized_timestamp


# --- HARDCODED CAMERA CALIBRATION MATRICES ---
K = np.array([
    [2294.54177, 0.0, 1969.36277],
    [0.0, 2307.85097, 1170.62330],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coefs = np.array([-0.39308027, 0.17945339, -0.00135962, -0.00164354, -0.03857585], dtype=np.float32)

# =========================================================================
# 3. GLOBAL THRESHOLDS & DRAGGABLE Y_EVAL
# =========================================================================
LANE_MERGE_DIST = int(v_width * 0.05)           
LANE_TRACK_DIST = int(v_width * 0.20)           
MIN_LANE_PIXELS = max(20, int(v_height * 0.02)) 

# --- HARDCODED DRAGGABLE EVALUATION LINE ---
y_eval = 1800

print("\n[SETUP CALIBRATION COMPLETE]")
print("-> Draggable Lane Width Evaluation Line Active.")
print("="*50 + "\n")

IRC_MIN_HEIGHT = 2.0                      
DESIGN_SPEED_KMH = 80

CLASS_NAMES = {
    0: 'Centre Lane Marking', 1: 'Chevron', 2: 'Kerb', 
    3: 'Rigid Crash Barrier', 4: 'Shoulder Marking', 5: 'Traffic Sign Damaged', 
    6: 'Traffic Sign Not Visible', 7: 'W-Beam Crash Barrier', 8: 'Mandatory Sign',
    9:'Informative Sign', 10: 'Warning Sign'
}

LANE_CLASSES = {0, 4}               
SIGN_CLASSES = {1 , 5 , 6 , 8 , 9 , 10}      
BOUNDARY_CLASSES = {2,7,3}           

LANE_COLORS = {0: (255, 200, 0), 4: (200, 0, 255)}
BOUNDARY_COLORS = {2: (0, 165, 255), 7: (200, 0, 255), 3: (0, 255, 255)}
BRIDGE_COLORS = [(0, 255, 0), (255, 0, 0), (0, 215, 255), (255, 0, 255), (0, 165, 255)]

# =========================================================================
# 4. DATA EXPORT CONFIGURATION
# =========================================================================
csv_log_path = os.path.join(args.output_dir, f"{_stem}_log.csv")
csv_file = open(csv_log_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp (sec)", "Expected Total Lanes", "Centre Lanes Detected", 
    "Shoulder Lanes Detected", "Lane Integrity Status", "Kerb Available", 
    "Crash Barrier Available", "Radius of Curvature (m)", "Curvature Severity",
    "Tracked Lane Widths (m)", "Traffic Signs [ID|Class|Height|Dist]"
])

# =========================================================================
# 5. MODEL INITIALIZATION
# =========================================================================
print("[INFO] Loading YOLO Models (Isolated Instances)...")
# Load two distinct instances so their internal memory states do not clash
master_model_lanes = YOLO(args.yolo_weights) 
master_model_signs = YOLO(args.yolo_weights) 
master_model=YOLO(args.yolo_weights) 



print("[INFO] Loading Depth Anything V2...")
model_configs = {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
depth_model = DepthAnythingV2(**{**model_configs, 'max_depth': 80})
depth_model.load_state_dict(torch.load(args.depth_checkpoint, map_location='cpu', weights_only=True))
depth_model = depth_model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

# =========================================================================
# 6. CALIBRATION HELPERS & GLOBALS
# =========================================================================
src_points_bev = []
dragging_idx = -1
p2_points = []
p2_dragging_idx = -1
p3_left_pct = 0.35
p3_right_pct = 0.65
p3_dragging_left = False
p3_dragging_right = False
p3_bev_points = []
p3_bev_dragging_idx = -1

y_eval_mppx = 1650
mppx_dragging = False

def extract_and_save_frame(video_path, timestamp, filename):
    os.makedirs("frames", exist_ok=True)
    frame_path = os.path.join("frames", filename)
    
    if os.path.exists(frame_path):
        try: os.remove(frame_path)
        except OSError: pass

    cap_f = cv2.VideoCapture(video_path)
    cap_f.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap_f.read()
    cap_f.release()
    
    if ret:
        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return cv2.imread(frame_path)
    else:
        print(f"[ERROR] Could not extract frame asset at {timestamp} seconds.")
        sys.exit()

def select_points(event, x, y, flags, param):
    global src_points_bev
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points_bev) < 4:
            src_points_bev.append([x, y])
            print(f"Captured Point {len(src_points_bev)}: [{x}, {y}]")

def phase_1_get_m_matrix(video_path, start_time, num_lanes):
    global src_points_bev
    src_points_bev = []  
    
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        print("\n" + "="*50)
        print("[PHASE 1] BEV M-MATRIX SELECTION")
        print("1. Interactive selection (Click 4 points on GUI)")
        print("2. Provide Pre-calculated hardcoded 3x3 M-Matrix directly")
        choice = input("Select an option [1/2]: ").strip()
        if choice in ['1', '2']: break
        
    if choice == '2':
        cap.release()  # Close the capture early since we don't need a frame
        print("\n" + "-"*50)
        print("PASTE YOUR FULL 3x3 MATRIX BLOCK BELOW:")
        print("(You can copy-paste the whole thing with brackets and commas directly)")
        print("-"*50)
        try:
            all_numbers = []
            
            # Continuously read incoming lines until we collect at least 9 values
            while len(all_numbers) < 9:
                line = input()
                if not line.strip() and len(all_numbers) == 0:
                    continue  # Skip leading empty entries if any
                
                # Strip out formatting brackets and commas, leaving only numbers and spaces
                cleaned = line.replace('[', ' ').replace(']', ' ').replace(',', ' ')
                
                # Extract valid numbers from the tokens
                for token in cleaned.split():
                    try:
                        all_numbers.append(float(token))
                    except ValueError:
                        pass  # Ignore any non-numeric strings
            
            if len(all_numbers) < 9:
                raise ValueError("Could not find 9 numbers in the provided input.")
                
            # Take the first 9 values and reshape them into a perfect 3x3 NumPy matrix
            matrix_data = all_numbers[:9]
            M = np.array(matrix_data, dtype=np.float32).reshape(3, 3)
            
            print("\n[SUCCESS] Manually pasted BEV M-Matrix loaded successfully!")
            
            # --- PRINT THE MATRIX ---
            print("\n" + "="*40)
            print("YOUR MANUALLY ENTERED M-MATRIX (3x3):")
            print("="*40)
            for row in M:
                print(f"  [ {row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f} ]")
            print("="*40)
            
            return M, orig_w, orig_h, fps
            
        except Exception as e:
            print(f"\n[ERROR] Invalid matrix values entered ({e}).")
            print("Falling back to Option 1 (Interactive selection)...")
            choice = '1'
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame at the specified timestamp.")
        exit()
        
    undistorted = cv2.undistort(frame, K, dist_coefs)

    window_name = "Phase 1: Select 4 Points for BEV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1920, 1080) 
    cv2.setMouseCallback(window_name, select_points)
    
    print("\n" + "="*50)
    print("[PHASE 1] BEV MATRIX SELECTION")
    if num_lanes in [2, 3]:
        print("!!! ALERT: For 2 or 3 lane roads, please SELECT THE ENTIRE ROAD (Shoulder to Shoulder).")
    else:
        print("For 4+ lanes, select the area capturing your required perspective.")
        
    print("Click 4 points in order: 1. Top-Left -> 2. Top-Right -> 3. Bottom-Right -> 4. Bottom-Left")
    print("Press 'r' to Reset points. Press 'ENTER' when done.")
    print("="*50)
    
    while True:
        img_display = undistorted.copy()
        for i, pt in enumerate(src_points_bev):
            cv2.circle(img_display, tuple(pt), 8, (0, 0, 255), -1)
            labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
            cv2.putText(img_display, labels[i], (pt[0] + 10, pt[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(src_points_bev) == 4: break
        elif key in [ord('r'), ord('R')]: src_points_bev = []
        elif key in [27, ord('q')]: exit()

    cv2.destroyAllWindows()
    
    src_4k = np.float32(src_points_bev)
    offset = int(orig_w * 0.25)
    dst_4k = np.float32([[offset, 0], [orig_w - offset, 0], [orig_w - offset, orig_h], [offset, orig_h]])
    
    M = cv2.getPerspectiveTransform(src_4k, dst_4k)
    print("\n[SUCCESS] BEV M-Matrix generated successfully!")
    # --- PRINT THE MATRIX ---
    print("\n" + "="*40)
    print("YOUR COMPUTED M-MATRIX (3x3):")
    print("="*40)
    for row in M:
        print(f"  [ {row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f} ]")
    print("="*40)
    return M, orig_w, orig_h, fps
    

def phase_2_get_mpp_x(video_path, M, width_4k, height_4k, synchronized_timestamp, num_lanes):
    print("\n" + "="*50)
    print("[PHASE 2] CALIBRATE GLOBAL MPP_X (Meters Per Pixel X)")
    print("Select your preferred calculation method:")
    print("  1. Manual Entry (Type a known value)")
    print("  2. Auto-Calculate (Average from video over 5 seconds)")
    print("  3. Frame Selection (Click 2 points on a specific frame)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        mpp_x = float(input("Enter your known global mpp_x value (e.g., 0.007): "))
        print(f"[SUCCESS] Global mpp_x set to: {mpp_x:.6f}")
        return mpp_x
        
    elif choice == '2':
        print("\n[INFO] Running Auto-Calculation (Current Method)...")
        total_width_meters = float(input("Enter total cross-sectional road width (meters): "))
        expected_lines = int(input("Enter expected number of lines crossed (e.g., 3 lanes = 4 lines): "))
        
        # We reuse the master_model loaded earlier in the script
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, synchronized_timestamp * 1000) 
        
        cv2.namedWindow('MPP_X Calibration (Press Q to finish)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MPP_X Calibration (Press Q to finish)', 1280, 720)
        
        scale_pool = []
        EVAL_Y = int(height_4k * 0.750) 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Using K and dist_coefs defined in your main script
            undistorted = cv2.undistort(frame, K, dist_coefs)
            results = master_model(undistorted, verbose=False)
            ai_points = []
            
            if results[0].masks is not None:
                # Iterate through masks and their corresponding class labels
                for idx, mask_pts in enumerate(results[0].masks.xy):
                    cls_id = int(results[0].boxes.cls[idx])
                    
                    # FILTER BASED ON data (1).yaml
                    # 0: Centre Lane Marking, 5: Shoulder Marking
                    if cls_id in [0, 5]:
                        valid_pts = [pt for pt in mask_pts if abs(pt[1] - EVAL_Y) < 40]
                        if len(valid_pts) > 0:
                            avg_x = sum(pt[0] for pt in valid_pts) / len(valid_pts)
                            ai_points.append((int(avg_x), EVAL_Y))
                        
            ai_points.sort(key=lambda pt: pt[0])
            
            if len(ai_points) == expected_lines:
                left_shoulder, right_shoulder = ai_points[0], ai_points[-1]
                outer_pts = np.array([left_shoulder, right_shoulder], dtype=np.float32).reshape(-1, 1, 2)
                warped_outer = cv2.perspectiveTransform(outer_pts, M)
                
                pixel_delta = abs(warped_outer[1][0][0] - warped_outer[0][0][0])
                if pixel_delta > 0:
                    scale_pool.append(total_width_meters / pixel_delta)
                    
                    cv2.line(undistorted, left_shoulder, right_shoulder, (0, 215, 255), 4)
                    cv2.circle(undistorted, left_shoulder, 8, (0, 0, 255), -1)
                    cv2.circle(undistorted, right_shoulder, 8, (0, 0, 255), -1)
                    current_avg = sum(scale_pool) / len(scale_pool)
                    cv2.putText(undistorted, f"Calibrating MPP_X: {current_avg:.6f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow('MPP_X Calibration (Press Q to finish)', undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyWindow('MPP_X Calibration (Press Q to finish)')
        
        # Calculates final average, defaulting to 0.007 if no frames were valid
        final_mpp_x = sum(scale_pool) / len(scale_pool) if scale_pool else 0.007
        print(f"\n[SUCCESS] Final Computed MPP_X: {final_mpp_x:.7f}")
        return final_mpp_x
        
    elif choice == '3':
        ts = float(input("Enter timestamp (in seconds) to extract frame: "))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[ERROR] Could not read frame at that timestamp.")
            return 0.01 # Fallback
            
        # --- THE FIX: UNDISTORT AND CONVERT TO BEV FIRST ---
        undistorted_frame = cv2.undistort(frame, K, dist_coefs)
        # Transform the image into BEV space at the full 4K resolution
        bev_frame = cv2.warpPerspective(undistorted_frame, M, (width_4k, height_4k))
        
        # Ensure frames directory exists and save the 4K BEV frame
        os.makedirs("frames", exist_ok=True)
        frame_path = f"frames/mpp_x_bev_calib_{ts}s.jpg"
        cv2.imwrite(frame_path, bev_frame)
        print(f"[INFO] 4K BEV Frame saved to {frame_path}")
        
        # Setup GUI for 1080p display to fit on screen
        display_w, display_h = 1920, 1080
        scale_x = width_4k / display_w
        scale_y = height_4k / display_h
        
        gui_points = []
        
        def select_pts(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(gui_points) < 2:
                    gui_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN: # Right click to undo
                if gui_points:
                    gui_points.pop()
                    
        window_name = "Select 2 Points for MPP_X (on BEV)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_w, display_h)
        cv2.setMouseCallback(window_name, select_pts)
        
        while True:
            # --- USE THE BEV FRAME FOR THE GUI ---
            display_frame = cv2.resize(bev_frame.copy(), (display_w, display_h))
            cv2.putText(display_frame, "Left-Click: Place Point | Right-Click: Undo | Press 'c': Confirm | Press 'r': Reset", 
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            for pt in gui_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            
            if len(gui_points) == 2:
                cv2.line(display_frame, gui_points[0], gui_points[1], (0, 255, 0), 2)
                
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                gui_points.clear()
            elif key == ord('c') and len(gui_points) == 2:
                break
                
        cv2.destroyWindow(window_name)
        
        # Convert 1080p clicks back up to 4K BEV coordinates
        pt1_4k_bev = np.array([gui_points[0][0] * scale_x, gui_points[0][1] * scale_y], dtype=np.float32)
        pt2_4k_bev = np.array([gui_points[1][0] * scale_x, gui_points[1][1] * scale_y], dtype=np.float32)
        
        # NOTE: Since the image was ALREADY converted to BEV, we do NOT use cv2.perspectiveTransform here.
        # We just calculate the Euclidean distance directly between the two 4K BEV points.
        bev_pixel_dist = np.linalg.norm(pt2_4k_bev - pt1_4k_bev)
        
        print(f"\n[INFO] Distance in 4K BEV pixels: {bev_pixel_dist:.2f} px")
        real_dist = float(input("Enter the real-world distance between these points in meters: "))
        
        mpp_x = real_dist / bev_pixel_dist
        print(f"[SUCCESS] Calculated global mpp_x: {mpp_x:.6f}")
        return mpp_x
        
    else:
        print("[ERROR] Invalid choice. Defaulting to 0.007")
        return 0.007
        
def p3_drag_lines(event, x, y, flags, param):
    global p3_left_pct, p3_right_pct, p3_dragging_left, p3_dragging_right
    orig_w = param  
    hitbox = 150 
    
    current_left_x = int(orig_w * p3_left_pct)
    current_right_x = int(orig_w * p3_right_pct)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(x - current_left_x) < hitbox: p3_dragging_left = True
        elif abs(x - current_right_x) < hitbox: p3_dragging_right = True
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if p3_dragging_left: p3_left_pct = max(0.02, min(x / orig_w, p3_right_pct - 0.05))
        elif p3_dragging_right: p3_right_pct = max(p3_left_pct + 0.05, min(x / orig_w, 0.98))
            
    elif event == cv2.EVENT_LBUTTONUP:
        p3_dragging_left = False
        p3_dragging_right = False

def p3_bev_drag_points(event, x, y, flags, param):
    global p3_bev_points, p3_bev_dragging_idx
    hit_radius = 40
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(p3_bev_points):
            if math.hypot(pt[0] - x, pt[1] - y) < hit_radius:
                p3_bev_dragging_idx = i
                return
    elif event == cv2.EVENT_MOUSEMOVE:
        if p3_bev_dragging_idx != -1: p3_bev_points[p3_bev_dragging_idx] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        p3_bev_dragging_idx = -1

def phase_3_get_mpp_y(video_path, M, orig_w, orig_h, start_time):
    global p3_left_pct, p3_right_pct, p3_bev_points
    
    while True:
        print("\n" + "="*50)
        print("[PHASE 3] MPP_Y SELECTION")
        print("1. Provide Pre-calculated hardcoded MPP_Y value directly")
        print("2. Calculate Dynamically using SAM 3")
        print("3. Calculate using one image after converting into BEV")
        choice = input("Select an option [1/2/3]: ").strip()
        if choice in ['1', '2', '3']: break
    
    if choice == '1':
        return float(input("Enter your known MPP_Y value (e.g., 0.0450): "))
        
    REAL_STRIP_LENGTH_M = float(input("\nEnter the real-world length of a single centre lane strip (in meters, e.g., 3.0): "))
        
    if choice == '3':
        ts = float(input("Enter timestamp (seconds) to extract frame for MPP_Y: "))
        raw_img = extract_and_save_frame(video_path, ts, "mppy_frame.jpg")
        undistorted = cv2.undistort(raw_img, K, dist_coefs)
        bev_img = cv2.warpPerspective(undistorted, M, (orig_w, orig_h))
        
        p3_bev_points = [[int(orig_w * 0.5), int(orig_h * 0.4)], [int(orig_w * 0.5), int(orig_h * 0.6)]]
        
        window_name = 'Phase 3 - Drag Points on BEV Image for Lane Strip Length. Press ENTER.'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 1080)
        cv2.setMouseCallback(window_name, p3_bev_drag_points)
        
        final_mpp_y = 0.0450
        
        while True:
            display_img = bev_img.copy()
            cv2.line(display_img, tuple(p3_bev_points[0]), tuple(p3_bev_points[1]), (255, 0, 255), 2)
            for pt in p3_bev_points: cv2.circle(display_img, tuple(pt), 6, (0, 0, 255), -1)
                
            pixel_delta = abs(p3_bev_points[1][1] - p3_bev_points[0][1])
            if pixel_delta > 0: final_mpp_y = REAL_STRIP_LENGTH_M / pixel_delta
                
            cv2.putText(display_img, f"Computed MPP_Y: {final_mpp_y:.6f}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow(window_name, display_img)
            if cv2.waitKey(30) & 0xFF == 13: break
            
        cv2.destroyWindow(window_name)
        return final_mpp_y

    print("\n[PHASE 3] MPP_Y Calibration Initialized via SAM 3.")
    
    sam3_overrides = dict(
        model=SAM3_MODEL_PATH,
        task="segment",
        mode="predict",
        device=0,
        imgsz=640,
        verbose=False
    )
    predictor = SAM3SemanticPredictor(overrides=sam3_overrides)
    
    cap_y = cv2.VideoCapture(video_path)
    cap_y.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000) 
    
    window_name = 'SAM 3 MPP_Y Calibration (Press Q to finish)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, p3_drag_lines, orig_w)
    
    collected_pixel_heights = []
    frame_counter = 0

    while cap_y.isOpened():
        ret, frame = cap_y.read()
        if not ret: break
        
        frame_counter += 1
        if frame_counter % 5 != 0: continue
        
        undistorted = cv2.undistort(frame, K, dist_coefs)
        roi_min_y, roi_max_y = int(orig_h * 0.45), int(orig_h * 0.95)
        roi_frame = undistorted[roi_min_y:roi_max_y, :]
        
        scale_factor = 0.5  
        small_roi = cv2.resize(roi_frame, (0, 0), fx=scale_factor, fy=scale_factor)
        
        left_boundary = int(orig_w * p3_left_pct)
        right_boundary = int(orig_w * p3_right_pct)
        
        cv2.line(undistorted, (left_boundary, roi_min_y), (left_boundary, roi_max_y), (0, 255, 0), 4)
        cv2.line(undistorted, (right_boundary, roi_min_y), (right_boundary, roi_max_y), (0, 255, 0), 4)
        cv2.putText(undistorted, "DRAG GREEN LINES TO ISOLATE DASHED LANE", (left_boundary - 200, roi_min_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        predictor.set_image(small_roi)
        results = predictor(text=["white dashed lane line"])
        
        if results and results[0].masks is not None:
            for mask_pts in results[0].masks.xy:
                if len(mask_pts) < 4: continue
                
                m_arr = np.array(mask_pts, dtype=np.float32)
                m_arr /= scale_factor
                c_x = int(np.mean(m_arr[:, 0]))
                
                if c_x < left_boundary or c_x > right_boundary: continue
                if (np.max(m_arr[:, 1]) - np.min(m_arr[:, 1])) > ((roi_max_y - roi_min_y) * 0.95): continue

                m_arr[:, 1] += roi_min_y  
                pts_to_warp = m_arr.reshape(-1, 1, 2)
                warped_pts = cv2.perspectiveTransform(pts_to_warp, M).reshape(-1, 2)
                
                pixel_height = abs(np.max(warped_pts[:, 1]) - np.min(warped_pts[:, 1]))
                
                if 10 < pixel_height < 1500:
                    collected_pixel_heights.append(pixel_height)
                    cv2.fillPoly(undistorted, [m_arr.astype(np.int32)], (255, 255, 0))

        torch.cuda.empty_cache()

        if len(collected_pixel_heights) > 0:
            sorted_heights = sorted(collected_pixel_heights)
            trim_idx = len(sorted_heights) // 4
            if len(sorted_heights) > 4:
                safe_heights = sorted_heights[trim_idx:-trim_idx]
            else:
                safe_heights = sorted_heights
                
            avg_pixels = sum(safe_heights) / len(safe_heights)
            
            if avg_pixels > 0:
                current_mpp_y = REAL_STRIP_LENGTH_M / avg_pixels
                cv2.putText(undistorted, f"Avg MPP_Y: {current_mpp_y:.6f}", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 4)

        cv2.imshow(window_name, undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_y.release()
    cv2.destroyAllWindows()
    
    if len(collected_pixel_heights) > 0:
        sorted_heights = sorted(collected_pixel_heights)
        trim_idx = len(sorted_heights) // 4
        
        if len(sorted_heights) > 4:
            safe_heights = sorted_heights[trim_idx:-trim_idx]
        else:
            safe_heights = sorted_heights
            
        final_avg_pixels = sum(safe_heights) / len(safe_heights)
        final_mpp_y = REAL_STRIP_LENGTH_M / final_avg_pixels if final_avg_pixels > 0 else 0.05
    else:
        final_mpp_y = 0.05
    
    print(f"\n[SUCCESS] Final Computed Avg MPP_Y: {final_mpp_y:.7f}")
    return final_mpp_y

# Run Calibration sequence
print("\n" + "="*50)
print("RUNNING BEV & METRIC CONVERSION CALIBRATION")
print("="*50)
M_matrix, width_4k, height_4k, video_fps = phase_1_get_m_matrix(input_video, CALIBRATION_SYNC_TS,EXPECTED_LANES)
final_mpp_x=phase_2_get_mpp_x(input_video, M_matrix, width_4k, height_4k, synchronized_timestamp, EXPECTED_LANES)
final_mpp_y = phase_3_get_mpp_y(input_video, M_matrix, width_4k, height_4k, CALIBRATION_SYNC_TS)
print("="*50 + "\n")

# =========================================================================
# 7. TRACKING VARIABLES & UI FUNCTIONS
# =========================================================================
lane_histories = {}
active_lines = {}
next_line_id = 0
curvature_history = []
smoothed_cy_dynamic = None 
sign_height_history = defaultdict(list)
sign_physical_heights = defaultdict(list)
frame_count = 0
main_dragging = False


def draw_elegant_label(frame, box, text, color=(180, 180, 180)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def draw_sign_ui(frame, box, sign_id, dist, gamma_deg, height_m, s_class, is_safe):
    x1, y1, x2, y2 = map(int, box)
    bg_color = (0, 180, 0) if is_safe else (0, 0, 200)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, 3)
    lines = [f"ID: {sign_id} | Dist: {dist:.1f}m", f"Angle (gamma): {gamma_deg:.1f} deg", f"Bottom Height: {height_m:.2f}m", f"Class: {s_class}"]
    max_w = max([cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for l in lines])
    pad = 12
    box_w = max_w + pad*2
    box_h = len(lines) * 28 + pad*2
    bx1, by2 = x1, y1 - 10
    bx2, by1 = bx1 + box_w, by2 - box_h
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
    curr_y = by1 + pad + 18
    for l in lines:
        cv2.putText(frame, l, (bx1 + pad, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        curr_y += 28

def draw_advanced_hud(frame, x, y, max_default_width, title, info_lines, highlight_text, highlight_color, footer_text):
    tw = cv2.getTextSize(title.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    hw = cv2.getTextSize(highlight_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0] + 40
    fw = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    max_line_w = 0
    for text, _ in info_lines:
        lw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        if lw > max_line_w:
            max_line_w = lw
    width = max(tw, hw, fw, max_line_w, max_default_width) + 40

    base_h, line_spacing, h_hgt, footer_h = 75, 35, 55, 45
    total_h = base_h + (len(info_lines) * line_spacing) + h_hgt + footer_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + total_h), (40, 45, 50), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + total_h), (100, 100, 110), 1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.putText(frame, title.upper(), (x + 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
    curr_y = y + 85
    for text, color in info_lines:
        cv2.putText(frame, text, (x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        curr_y += line_spacing
    hx, hy, hw_box = x + 20, curr_y, width - 40
    cv2.rectangle(frame, (hx, hy), (hx + hw_box, hy + h_hgt), highlight_color, -1)
    text_size = cv2.getTextSize(highlight_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
    text_x, text_y = hx + (hw_box - text_size[0]) // 2, hy + (h_hgt + text_size[1]) // 2 - 3
    cv2.putText(frame, highlight_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    curr_y += h_hgt + 30
    cv2.putText(frame, footer_text, (x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

def draw_lane_integrity_hud(frame, x, y, center_count, shoulder_count, exp_c, exp_s, has_barrier, barrier_detected, has_kerb, kerb_detected):
    missing_centers = max(0, exp_c - center_count)
    missing_shoulders = max(0, exp_s - shoulder_count)
    
    missing_items = []
    if missing_shoulders > 0:
        missing_items.append(f"{missing_shoulders} Shoulder(s)")
    if missing_centers > 0:
        missing_items.append(f"{missing_centers} Center(s)")
    if has_barrier and not barrier_detected:
        missing_items.append("Crash Barrier")
    if has_kerb and not kerb_detected:
        missing_items.append("Kerb")
        
    title = "ROAD FEATURES & INTEGRITY MONITOR"
    
    if not missing_items:
        status_text = "STATUS: All Features Intact"
        status_color = (0, 255, 0)
        sub_text = f"Monitoring {center_count + shoulder_count} Lanes & Expected Boundaries."
    else:
        status_text = "MISSING: " + ", ".join(missing_items)
        status_color = (0, 165, 255) if (center_count > 0 or shoulder_count > 0) else (0, 0, 255)
        sub_text = f"Expected: {exp_c}C, {exp_s}S | Barrier: {'Yes' if has_barrier else 'No'}, Kerb: {'Yes' if has_kerb else 'No'}"

    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    sw = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
    subw = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    
    width = max(tw, sw, subw) + 40
    height = 140
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (45, 40, 40), -1) 
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 200, 0), 3) 
    
    cv2.putText(frame, title, (x + (width - tw)//2, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA) 
    cv2.putText(frame, status_text, (x + 20, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, sub_text, (x + 20, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return status_text

# Callback to dynamically drag the width evaluation line up and down during video inference!
def main_inference_drag(event, x, y, flags, param):
    global y_eval, main_dragging
    hitbox = 40
    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(y - y_eval) < hitbox:
            main_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if main_dragging:
            y_eval = y
    elif event == cv2.EVENT_LBUTTONUP:
        main_dragging = False

# =========================================================================
# 8. MAIN PROCESSING LOOP
# =========================================================================
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (v_width, v_height))
out_bev = cv2.VideoWriter(output_bev_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (v_width, v_height))

cv2.namedWindow('Unified ADAS Audit Pipeline', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Unified ADAS Audit Pipeline', main_inference_drag)

# --- Dynamic Tripwire & Tracking Initialization (From New_code) ---
radius_history = deque(maxlen=30)
pt_native = np.array([[[v_width / 2.0, y_eval]]], dtype=np.float32)
global_tripwire_bev_y = cv2.perspectiveTransform(pt_native, M_matrix)[0][0][1]
lane_width_histories = []
car_center_x_pixel = v_width / 2.0
global_tripwire_bev_y = 0
smoothed_R = 5000.0
is_curve_safe = True
# --- FOOLPROOF TRIPWIRE INITIALIZATION ---
# 1. Create a blank image and draw the y_eval line
dummy_mask = np.zeros((v_height, v_width), dtype=np.uint8)
cv2.line(dummy_mask, (0, y_eval), (v_width, y_eval), 255, 2)

# 2. Warp it into BEV just like the lanes
bev_dummy = cv2.warpPerspective(dummy_mask, M_matrix, (v_width, v_height))

# 3. Find exactly where the line landed in the BEV image
init_y_coords, _ = np.where(bev_dummy > 127)

if len(init_y_coords) > 0:
    global_tripwire_bev_y = float(np.max(init_y_coords))
else:
    global_tripwire_bev_y = v_height * 0.85  # Safe fallback if out of bounds
tripwire_locked = False
overlap_y_mins = []
overlap_y_maxs = []
print(f"[INFO] Started with hardcoded native y_eval: {y_eval} -> BEV Y: {global_tripwire_bev_y:.1f}px")

# Fallback/Safety initialization for threshold variables if not declared globally earlier
if 'radius_threshold' not in locals() and 'radius_threshold' not in globals():
    radius_threshold = radius_threshold if 'radius_threshold' in locals() or 'radius_threshold' in globals() else 400.0
if 'WIDTH_THRESHOLD' not in locals() and 'WIDTH_THRESHOLD' not in globals():
    WIDTH_THRESHOLD = width_threshold if 'width_threshold' in locals() or 'width_threshold' in globals() else 3.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, dist_coefs, (v_width, v_height), 0, (v_width, v_height))
    undistorted = cv2.undistort(frame, K, dist_coefs, None, new_camera_matrix)
    
    clean_undistorted = undistorted.copy() 
    
    fy_current, cy_optical, cx_optical = float(new_camera_matrix[1,1]), float(new_camera_matrix[1,2]), float(new_camera_matrix[0,2])

    rgb_frame = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            depth_map = depth_model.infer_image(rgb_frame)

    # =======================================================
    # [FIXED] TWO-PASS EXECUTION WITH ISOLATED MEMORY
    # =======================================================
    # Pass 1: Predict lanes & boundaries without tracking
    lane_results = master_model_lanes.predict(undistorted, verbose=False, classes=[0,2,3,4,7])

    # Pass 2: Track ONLY signs using BoT-SORT
    sign_results = master_model_signs.track(undistorted, persist=True, tracker="botsort.yaml", verbose=False, conf=0.05, classes=[1,5,6,8,9,10])
    
    lanes_data, signs_data, boundary_data, boundary_masks_data = [], [], [], []
    lane_overlay = undistorted.copy()

    # --- 1. PROCESS LANES AND BOUNDARIES (From lane_results) ---
    if lane_results[0].masks is not None and lane_results[0].boxes is not None:
        for mask_xy, cls in zip(lane_results[0].masks.xy, lane_results[0].boxes.cls):
            c = int(cls)
            poly = np.array(mask_xy, dtype=np.int32)
            binary_mask = np.zeros((v_height, v_width), dtype=np.uint8)
            cv2.fillPoly(binary_mask, [poly], 1)
            
            if c in LANE_CLASSES:
                y_idx, x_idx = np.where(binary_mask == 1)
                if len(y_idx) > 50:
                    med_x = np.median(x_idx)
                    lanes_data.append({'mask': binary_mask, 'y': y_idx, 'x': x_idx, 'med_x': med_x, 'class_name': CLASS_NAMES[c], 'cls_id': c})
                    
                    mask_color = (255, 180, 0) if c == 0 else (0, 120, 255)
                    cv2.fillPoly(lane_overlay, [poly], mask_color)
                    
            elif c in BOUNDARY_CLASSES:
                boundary_masks_data.append({'mask': binary_mask, 'cls_id': c})
                
                b_color = BOUNDARY_COLORS.get(c, (0, 255, 255))
                cv2.fillPoly(lane_overlay, [poly], b_color)

    if lane_results[0].boxes is not None:
        for box, cls in zip(lane_results[0].boxes.xyxy, lane_results[0].boxes.cls):
            c = int(cls)
            if c in BOUNDARY_CLASSES:
                bx = box.cpu().numpy()
                boundary_data.append({'box': bx, 'class_name': CLASS_NAMES[c], 'cls_id': c})
                
                x1, y1, x2, y2 = map(int, bx)
                b_color = BOUNDARY_COLORS.get(c, (0, 165, 255))
                cv2.rectangle(undistorted, (x1, y1), (x2, y2), b_color, 3)
                cv2.putText(undistorted, CLASS_NAMES[c], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, b_color, 2)

    # --- 2. PROCESS TRACKED SIGNS (From sign_results) ---
    if sign_results[0].boxes is not None:
        track_ids = sign_results[0].boxes.id
        if track_ids is None: track_ids = [None] * len(sign_results[0].boxes)
            
        for box, cls, t_id in zip(sign_results[0].boxes.xyxy, sign_results[0].boxes.cls, track_ids):
            c = int(cls)
            if c in SIGN_CLASSES and t_id is not None:
                signs_data.append({'box': box.cpu().numpy(), 'id': int(t_id), 'class_name': CLASS_NAMES[c]})

    # [FIX] Blend masks beautifully over the road
    cv2.addWeighted(lane_overlay, 0.4, undistorted, 0.6, 0, undistorted)


    # =======================================================
    # MODULE 1: ROAD CURVATURE & PITCH CALCULATION
    # =======================================================
    raw_cy = cy_optical
    left_lines, right_lines = [], []
    for lane in lanes_data:
        valid_mask = lane['y'] > (v_height * 0.5)
        y_near, x_near = lane['y'][valid_mask], lane['x'][valid_mask]
        if len(y_near) > 50:
            pts = np.column_stack((x_near, y_near)).astype(np.float32)
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            if abs(vx[0]) > 1e-6:
                m = vy[0] / vx[0]
                c = y0[0] - m * x0[0]
                if m < -0.1: left_lines.append((m, c))
                elif m > 0.1: right_lines.append((m, c))

    if left_lines and right_lines:
        avg_m_left, avg_c_left = np.mean([l[0] for l in left_lines]), np.mean([l[1] for l in left_lines])
        avg_m_right, avg_c_right = np.mean([l[0] for l in right_lines]), np.mean([l[1] for l in right_lines])
        if abs(avg_m_left - avg_m_right) > 1e-6:
            intersect_x = (avg_c_right - avg_c_left) / (avg_m_left - avg_m_right)
            intersect_y = (avg_m_left * intersect_x) + avg_c_left
            if 0 < intersect_y < v_height: raw_cy = float(intersect_y)

    smoothed_cy_dynamic = raw_cy if smoothed_cy_dynamic is None else (0.1 * raw_cy) + (0.9 * smoothed_cy_dynamic)
    pitch_rad = math.atan((smoothed_cy_dynamic - cy_optical) / fy_current)
    # --- ADD THIS: DRAW THE DYNAMIC HORIZON ---
    if smoothed_cy_dynamic is not None:
        horizon_y = int(smoothed_cy_dynamic)
        cv2.line(undistorted, (0, horizon_y), (v_width, horizon_y), (0, 255, 255), 2)
        cv2.putText(undistorted, "DYNAMIC HORIZON", (20, horizon_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


    # Dynamic alias linking to ensure calibration variables match your exact scope definitions
    M = M_matrix if 'M_matrix' in locals() or 'M_matrix' in globals() else M
    current_mpp_x = final_mpp_x if 'final_mpp_x' in locals() or 'final_mpp_x' in globals() else final_mpp_x
    current_mpp_y = final_mpp_y if 'final_mpp_y' in locals() or 'final_mpp_y' in globals() else final_mpp_y

    all_lines_data = []
    for lane in lanes_data:
        if lane['cls_id'] in [0, 4]:
            lane_mask = (lane['mask'] * 255).astype(np.uint8)
            bev_lane_mask = cv2.warpPerspective(lane_mask, M, (v_width, v_height))
            
            # Extract points from BEV mask
            y_coords, x_coords = np.where(bev_lane_mask > 127)
            if len(y_coords) > 50:
                A_px, B_px, C_px = np.polyfit(y_coords, x_coords, 2)
                center_px = np.polyval([A_px, B_px, C_px], v_height * 0.8) # Base evaluation
                
                all_lines_data.append({
                    "x_bev_t": x_coords,
                    "y_bev_t": y_coords,
                    "A_px": A_px,
                    "B_px": B_px,
                    "C_px": C_px,
                    "center_pixel": center_px,
                    "min_y": np.min(y_coords),
                    "max_y": np.max(y_coords)
                })

    # [FIX] Link matched_lines interface array to active BEV lines structure
    matched_lines = all_lines_data
    
    # [FIX] Initialize placeholder arrays for downstream HUD engines before condition checks
    width_info = None
    width_safe = True
    csv_widths = []

    # --- DYNAMIC TRIPWIRE LOCKING ---
    
    # 1. Start immediately with the hardcoded y_eval mapped to BEV
    if global_tripwire_bev_y is None:
        pt_native = np.array([[[v_width / 2.0, y_eval]]], dtype=np.float32)
        global_tripwire_bev_y = cv2.perspectiveTransform(pt_native, M)[0][0][1]
        print(f"[INFO] Starting with hardcoded native y_eval: {y_eval} -> BEV Y: {global_tripwire_bev_y:.1f}px")

    # 2. Gather data in the background to calculate the true dynamic tripwire
    if not tripwire_locked and len(all_lines_data) >= 2:
        all_lines_data_dist = sorted(all_lines_data, key=lambda x: abs(x["center_pixel"] - car_center_x_pixel))
        closest_ego = all_lines_data_dist[:2]
        
        y_min_overlap = max(closest_ego[0]["min_y"], closest_ego[1]["min_y"])
        y_max_overlap = min(closest_ego[0]["max_y"], closest_ego[1]["max_y"])
        
        if y_max_overlap > y_min_overlap + 50: 
            overlap_y_mins.append(y_min_overlap)
            overlap_y_maxs.append(y_max_overlap)
            
        if len(overlap_y_mins) >= 20: # required_cal_frames
            avg_min_y = np.mean(overlap_y_mins)
            avg_max_y = np.mean(overlap_y_maxs)
            
            # 3. Switch from hardcoded to dynamic!
            global_tripwire_bev_y = avg_max_y - ((avg_max_y - avg_min_y) * 0.15)
            tripwire_locked = True
            print(f"[SUCCESS] SWITCHED TO DYNAMIC TRIPWIRE AT BEV Y: {global_tripwire_bev_y:.1f}px")

    # --- RADIUS & WIDTH PROCESSING (RUNS ONCE TRIPWIRE IS LOCKED) ---
    if tripwire_locked and len(all_lines_data) > 0:
        
        # Evaluate pixel X positions at the locked horizontal plane
        for line in all_lines_data:
            line["x_eval_px"] = line["A_px"] * (global_tripwire_bev_y**2) + line["B_px"] * global_tripwire_bev_y + line["C_px"]
        
        # -- ROAD RADIUS (Isolate center ego lanes) --
        all_lines_data_dist = sorted(all_lines_data, key=lambda x: abs(x["center_pixel"] - car_center_x_pixel))
        closest_ego_lanes = all_lines_data_dist[:2]
        
        current_radii = []
        for line in closest_ego_lanes:
            x_m = line["x_bev_t"] * current_mpp_x
            y_m = line["y_bev_t"] * current_mpp_y
            A_m, B_m, C_m = np.polyfit(y_m, x_m, 2)
            eval_y_meters = global_tripwire_bev_y * current_mpp_y
            denom = np.absolute(2 * A_m)
            
            R = ((1 + (2 * A_m * eval_y_meters + B_m)**2)**1.5) / denom if denom > 1e-5 else 5000.0
            current_radii.append(min(R, 5000.0))
            
        if len(current_radii) > 0:
            avg_radius = np.mean(current_radii)
            radius_history.append(avg_radius)
            smoothed_R = np.mean(radius_history)
            
            # Check Curvature Severity based on your global radius_threshold
            if smoothed_R < radius_threshold:
                is_curve_safe = False
            else:
                is_curve_safe = True
        
       # =========================================================================
            # -- MULTI-LANE WIDTHS (Pure Pixel-to-Meter Logic from New_code.py) --
            # =========================================================================
            
            # Sort all lines from left to right based on their evaluation point
            all_lines_data.sort(key=lambda item: item["x_eval_px"])
            
            csv_widths = []
            width_info = []
            tripwire_visual_pts = []
            current_frame_lane_widths = []
            
            if len(all_lines_data) > 1:
                for i in range(len(all_lines_data) - 1):
                    # Ensure histories exist for every lane segment
                    while len(lane_width_histories) <= i:
                        lane_width_histories.append(deque(maxlen=30))
                    
                    # EXACT LOGIC FROM NEW_CODE.PY: Pure pixel distance * final_mpp_x
                    pixel_width_bev = abs(all_lines_data[i+1]["x_eval_px"] - all_lines_data[i]["x_eval_px"])
                    w = pixel_width_bev * final_mpp_x
                    
                    # Prevent zero-drops by using your established bounds
                    if 1.5 < w < 8.0:
                        lane_width_histories[i].append(w)
                        smoothed_w = np.median(lane_width_histories[i])
                        current_frame_lane_widths.append(smoothed_w)
                        csv_widths.append(f"{smoothed_w:.2f}")
                        
                        if smoothed_w < WIDTH_THRESHOLD:
                            width_safe = False
                            w_color = (0, 0, 255)
                        else:
                            w_color = (0, 255, 0)
                        
                        width_info.append((f"Lane {i+1} Width: {smoothed_w:.2f}m", w_color))
                        
                        x_bev_left = all_lines_data[i]["x_eval_px"]
                        x_bev_right = all_lines_data[i+1]["x_eval_px"]
                        
                        # Map BEV pixels back to Native 4K pixels for drawing
                        pts_bev = np.array([[[x_bev_left,  global_tripwire_bev_y], [x_bev_right,  global_tripwire_bev_y]]], dtype=np.float32)
                        M_inv = np.linalg.inv(M)
                        pts_4k = cv2.perspectiveTransform(pts_bev, M_inv)[0].astype(int)
                        
                        pt1, pt2 = tuple(pts_4k[0]), tuple(pts_4k[1])
                        tripwire_visual_pts.append((pt1, pt2, smoothed_w, i+1))

                # DRAW COLORFUL WIDTH LINES & TEXT ON THE ROAD
                LANE_COLORS = [
                    (0, 255, 255),   # L1: Yellow
                    (255, 144, 30),  # L2: Deep Sky Blue
                    (0, 255, 0),     # L3: Green
                    (255, 0, 255),   # L4: Magenta
                    (0, 165, 255),   # L5: Orange
                    (255, 0, 0)      # L6: Blue
                ]

                if len(tripwire_visual_pts) > 0:
                    # Cap visual rendering to expected lanes to avoid drawing ghost lines on shoulders
                    valid_visual_pts = tripwire_visual_pts[:EXPECTED_LANES]
                    
                    for idx, (pt1, pt2, smoothed_w, lane_idx) in enumerate(valid_visual_pts):
                        lane_color = LANE_COLORS[idx % len(LANE_COLORS)]
                        
                        # Draw visual tripwire segments beautifully aligned across the single track line
                        cv2.line(undistorted, pt1, pt2, lane_color, 4)
                        cv2.circle(undistorted, pt1, 8, lane_color, -1)
                        cv2.circle(undistorted, pt2, 8, lane_color, -1)
                        
                        # Center text inside its specific lane segment zone
                        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 - 20
                        cv2.rectangle(undistorted, (mid_x - 70, mid_y - 25), (mid_x + 70, mid_y + 10), lane_color, -1)
                        cv2.putText(undistorted, f"L{lane_idx}: {smoothed_w:.2f}m", (mid_x - 65, mid_y + 2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                width_info.append((f"IRC Min Target: {WIDTH_THRESHOLD:.2f}m", (220, 220, 220)))
    # =======================================================
    # MODULE 3: TRAFFIC SIGN CLEARANCES
    # =======================================================
    sign_info, csv_signs = [], []
    sign_safe = True
    for sign in signs_data:
        x1, y1, x2, y2 = map(int, sign['box'])
        t_id, s_class = sign['id'], sign['class_name']
        bbox_pixel_h = y2 - y1
        y_bottom = float(y2)
        
        # Sample depth slightly above the bottom-center of the bounding box
        sy, sx = int(y2 - 0.15 * (y2 - y1)), int((x1 + x2) / 2)
        
        # Ensure we don't go out of bounds on the depth map
        depth_patch = depth_map[max(0, sy - 2):min(v_height, sy + 3), max(0, sx - 2):min(v_width, sx + 3)]
        raw_d = float(np.median(depth_patch))
        
        if raw_d > 0 and not np.isnan(raw_d):
            # 1. Calibrated distance formula ported from the standalone model
            ai_distance = raw_d * 0.4
            
            # 2. Geometric distance locking (locks physical height after 15 frames)
            if len(sign_physical_heights[t_id]) < 15:
                phys_h = (bbox_pixel_h * ai_distance) / fy_current
                sign_physical_heights[t_id].append(phys_h)
                d = ai_distance 
            else:
                locked_phys_h = np.median(sign_physical_heights[t_id])
                d = (locked_phys_h * fy_current) / bbox_pixel_h
                
            # 3. Direct angle and height formulation using the dynamic horizon
            gamma_rad = math.atan((smoothed_cy_dynamic - y_bottom) / fy_current)
            h_raw = (d * math.tan(gamma_rad)) + camera_height_hc
            
            # 4. Smooth out the calculated height historically
            sign_height_history[t_id].append(h_raw)
            if len(sign_height_history[t_id]) > 10: 
                sign_height_history[t_id].pop(0)
            
            h_smoothed = float(np.median(sign_height_history[t_id]))
            
            # Determine safety based on the minimum threshold
            is_s_safe = h_smoothed >= IRC_MIN_HEIGHT
            if not is_s_safe: 
                sign_safe = False
            
            # Draw UI components and append to CSV logs
            draw_sign_ui(undistorted, sign['box'], t_id, d, math.degrees(gamma_rad), h_smoothed, s_class, is_s_safe)
            sign_info.append((f"Sign {t_id}: {h_smoothed:.1f}m", (220, 220, 220)))
            csv_signs.append(f"ID:{t_id}|{s_class}|H:{h_smoothed:.2f}m|D:{d:.1f}m")
    # =======================================================
    # UI RENDERING & EXCEL LOGGING
    # =======================================================
    curv_txt = ">800m (Straight)" if smoothed_R >= 800.0 else f"{smoothed_R:.1f} m"
    draw_advanced_hud(undistorted, v_width - 550, 40, 510, "ROAD CURVATURE SAFETY AUDIT",
        [(f"Design Speed limit: {DESIGN_SPEED_KMH} km/h", (220, 220, 220)), (f"IRC Safe Min Radius: {radius_threshold} m", (220, 220, 220)), (f"Est. Curvature (R): {curv_txt}", (0, 255, 0) if is_curve_safe else (0, 0, 255))],
        "STATUS: SAFE " if is_curve_safe else "STATUS: DANGEROUS CURVE",
        (0, 180, 0) if is_curve_safe else (0, 0, 200),
        f"Pitch: {math.degrees(pitch_rad):.2f} deg | Horiz: {int(smoothed_cy_dynamic)} px"
    )

    draw_advanced_hud(undistorted, 40, 40, 480, "LANE AUDIT",
        width_info if width_info else [("Scanning for lane markings...", (150, 150, 150))],
        "STATUS: WIDTH SAFE" if width_safe and width_info else "WARNING: NARROW LANES" if width_info else "STATUS: SEARCHING",
        (0, 180, 0) if width_safe and width_info else (0, 0, 200) if width_info else (100, 100, 100),
        f"Expected Lanes: {EXPECTED_LANES}"
    )

    active_lane_count = max(0, len(matched_lines) - 1)
    c_count = min(active_lane_count, EXPECTED_CENTERS) if active_lane_count > 0 else 0
    s_count = min(len(matched_lines) - c_count, 2) if len(matched_lines) > 0 else 0
    
    kerb_detected = any(b['cls_id'] == 2 for b in boundary_data) or any(bm['cls_id'] == 2 for bm in boundary_masks_data)
    barrier_detected = any(b['cls_id'] in [3, 7] for b in boundary_data) or any(bm['cls_id'] in [3, 7] for bm in boundary_masks_data)

    lane_status_log = draw_lane_integrity_hud(undistorted, v_width - 690, v_height - 180, c_count, s_count, EXPECTED_CENTERS, EXPECTED_SHOULDERS, HAS_CRASH_BARRIER, barrier_detected, HAS_KERB, kerb_detected)

    draw_advanced_hud(undistorted, 40, v_height - 350 - (len(sign_info)*35), 480, "SIGN CLEARANCE AUDIT",
        sign_info if sign_info else [("No active signs", (150, 150, 150))],
        "STATUS: CLEARANCES SAFE" if sign_safe and sign_info else "WARNING: LOW CLEARANCE" if sign_info else "STATUS: SEARCHING",
        (0, 180, 0) if sign_safe and sign_info else (0, 0, 200) if sign_info else (100, 100, 100),
        f"IRC Min Height: {IRC_MIN_HEIGHT}m"
    )

    # ... [End of all your HUD and calculation code] ...

    # 1. Write the frame to your output video file
    out.write(undistorted)
    
    # 2. [FIX] Display the annotated frame in the main window!
    cv2.imshow('Unified ADAS Audit Pipeline', cv2.resize(undistorted, (1280, 720)))
    # =======================================================
    # EXCEL LOGGING (0.1 Second Intervals)
    # =======================================================
    log_interval_frames = max(1, int(fps * 0.1))

    if frame_count % log_interval_frames == 0:
        current_time_sec = frame_count / fps
        kerb_avail = "YES" if kerb_detected else "NO"
        barrier_avail = "YES" if barrier_detected else "NO"
        curve_sev = "SAFE" if is_curve_safe else "DANGEROUS"
        width_str = " | ".join(csv_widths) if csv_widths else "None"
        sign_str = " | ".join(csv_signs) if csv_signs else "None"
        
        csv_writer.writerow([
            f"{current_time_sec:.1f}", EXPECTED_LANES, c_count, s_count, lane_status_log, 
            kerb_avail, barrier_avail, f"{smoothed_R:.2f}", curve_sev, 
            width_str, sign_str
        ])
    
    # 3. Optional: Display your BEV workspace window
    if GENERATE_BEV:
        bev_frame = cv2.warpPerspective(clean_undistorted, M_matrix, (v_width, v_height))
        cv2.imshow('2. BEV Workspace', cv2.resize(bev_frame, (800, 800)))
        out_bev.write(bev_frame)
        
    # 4. Standard window event handling loop
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
out.release()
out_bev.release()
csv_file.close() 
print(f"[SUCCESS] Fully scaled aesthetic Audit Rendered to: {output_video}")
print(f"[SUCCESS] Telemetry Log saved to: {csv_log_path}")