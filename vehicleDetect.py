# vehicleDetect.py
# OPTIMIZED MULTI-VARIABLE LIGHTING STATE CONTROLLER

import cv2
import numpy as np

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
LED_COLS = 8
GLARE_BRIGHTNESS_THRESHOLD = 235
GLARE_SYSTEM_THRESHOLD = 32.0        # Glare baseline evaluation index limit
DISTANCE_DIMMING_THRESHOLD = 110.0   # Optimized distance in meters to capture the target vehicle

REAL_VEHICLE_HEIGHT = 1.5
KNOWN_DISTANCE = 20.0
KNOWN_BOX_HEIGHT = 120.0

FOCAL_LENGTH = (KNOWN_BOX_HEIGHT * KNOWN_DISTANCE) / REAL_VEHICLE_HEIGHT

###############################################################################
# LIGHTING STATE MECHANICS
###############################################################################
tracked_objects = {}
next_object_id = 0
MAX_TRACKING_DISTANCE = 120

previous_distances = {}
current_brightness = np.ones(LED_COLS)
target_brightness = np.ones(LED_COLS)
SMOOTHING_FACTOR = 0.18

def update_smooth_brightness(dark_columns):
    global current_brightness, target_brightness
    target_brightness.fill(1.0)
    for col in dark_columns:
        if 0 <= col < LED_COLS:
            target_brightness[col] = 0.1
            
    current_brightness += SMOOTHING_FACTOR * (target_brightness - current_brightness)

def get_lane_based_dark_columns(w, cx):
    col_width = w / LED_COLS
    col_idx = int(cx // col_width)
    return {max(0, col_idx - 1), col_idx, min(LED_COLS - 1, col_idx + 1)}

###############################################################################
# NEURAL BOX SEGMENT VALIDATION REJECTION
###############################################################################
def is_valid_vehicle_detection(box, score, w, h, frame):
    if score < 0.45:  
        return False
        
    ymin, xmin, ymax, xmax = box
    x1, x2 = int(xmin * w), int(xmax * w)
    y1, y2 = int(ymin * h), int(ymax * h)

    if x1 >= x2 or y1 >= y2: 
        return False
        
    area = (x2 - x1) * (y2 - y1)
    if area < 450:  
        return False

    aspect = (y1 - y2) / (x1 - x2 + 1) if (x1 - x2) != 0 else 0
    if aspect > 0.95 or aspect < 0.2: 
        return False

    if (y1 + y2) / 2 < h * 0.18: 
        return False

    return True

###############################################################################
# DRAW MATRIX PANEL (CAR PRESERVED UNDERLAY)
###############################################################################
def draw_vehicle_boxes(frame, boxes, scores, valid_indices):
    h, w = frame.shape[:2]
    for i in valid_indices:
        if i >= len(boxes) or i >= len(scores):
            continue
            
        ymin, xmin, ymax, xmax = boxes[i]
        x1, x2 = int(xmin * w), int(xmax * w)
        y1, y2 = int(ymin * h), int(ymax * h)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        obj_id = get_object_id(cx, cy)
        dist = estimate_distance(boxes[i], h, obj_id)
        confidence = int(scores[i] * 100)
        
        label = f"Vehicle {obj_id} | {dist}m | {confidence}%"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

def draw_animated_matrix(frame, dark_columns):
    update_smooth_brightness(dark_columns)
    h, w = frame.shape[:2]

    matrix_h = 90
    matrix_w = 400
    margin = 10

    start_x = w - matrix_w - margin
    start_y = h - matrix_h - margin

    cell_w = matrix_w // LED_COLS
    cell_h = matrix_h // 8

    for row in range(8):
        for col in range(LED_COLS):
            x = start_x + col * cell_w
            y = start_y + row * cell_h
            cx = x + cell_w // 2
            cy = y + cell_h // 2
            radius = min(cell_w, cell_h) // 2 - 1

            brightness = current_brightness[col]
            blue = int(255 * brightness)
            green = int(255 * brightness)

            if row < 4:
                color = (0, green, blue)
            else:
                color = (0, int(180 * brightness), blue)

            cv2.circle(frame, (cx, cy), radius, color, -1)

###############################################################################
# HUD CONTROL WINDOW LAYER
###############################################################################
def draw_hud_telemetry(frame, frame_num, total_frames, num_vehicles, glare_score, current_fps, avg_fps, closest_distance=150.0):
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (0, 0), (280, 160), (20, 20, 20), -1)
    
    lines = [
        f"Frame: {frame_num}/{total_frames}",
        f"Vehicles: {num_vehicles}",
        f"Glare Score: {glare_score:.1f}",
        f"Current FPS: {current_fps:.2f}",
        f"Average FPS: {avg_fps:.2f}"
    ]
    
    y_offset = 25
    for line in lines:
        if "Current FPS" in line:
            color = (0, 255, 255)
        elif "Average FPS" in line:
            color = (0, 255, 0)
        else:
            color = (200, 200, 200)
            
        cv2.putText(frame, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_offset += 22

    # Global State Check: Glare limit breached AND target is inside optimized distance envelope
    if glare_score > GLARE_SYSTEM_THRESHOLD and closest_distance < DISTANCE_DIMMING_THRESHOLD and num_vehicles > 0:
        cv2.putText(frame, "HIGH BEAM INACTIVE", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Exact disclaimer alert requested formatting layout update
        banner_y = 200
        cv2.rectangle(frame, (15, banner_y), (250, banner_y + 45), (0, 0, 255), -1)
        cv2.putText(frame, "Headlight dimmed!", (25, banner_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "HIGH BEAM ACTIVE", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
    cv2.putText(frame, "AI-Assisted Lane-Aware Adaptive Lighting", (15, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

###############################################################################
# GEOMETRIC VECTOR CALCULATIONS
###############################################################################
def get_object_id(cx, cy):
    global tracked_objects, next_object_id
    best_id, best_dist = None, float('inf')

    for obj_id, (px, py) in tracked_objects.items():
        dist = abs(cx - px) + abs(cy - py)
        if dist < best_dist:
            best_dist = dist
            best_id = obj_id

    if best_dist < MAX_TRACKING_DISTANCE:
        tracked_objects[best_id] = (cx, cy)
        return best_id

    obj_id = next_object_id
    next_object_id += 1
    tracked_objects[obj_id] = (cx, cy)
    return obj_id

def estimate_distance(box, frame_height, object_id=0):
    global previous_distances
    ymin, xmin, ymax, xmax = box
    box_h = (ymax - ymin) * frame_height

    if box_h <= 0: return 150

    raw = (REAL_VEHICLE_HEIGHT * FOCAL_LENGTH) / box_h
    raw = max(3, min(raw, 150))

    if object_id not in previous_distances:
        previous_distances[object_id] = raw

    prev = previous_distances[object_id]
    smooth = 0.92 * prev + 0.08 * raw

    if smooth > prev:
        smooth = min(smooth, prev + 0.8)
    else:
        smooth = max(smooth, prev - 3.5)

    previous_distances[object_id] = smooth
    return int(smooth)

###############################################################################
# TARGET REGION ISOLATED GLARE ANALYSIS METHOD
###############################################################################
def compute_glare_score(frame, boxes, scores, valid):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vals = []

    for i in valid:
        if i >= len(boxes): continue
        ymin, xmin = int(boxes[i][0] * h), int(boxes[i][1] * w)
        ymax, xmax = int(boxes[i][2] * h), int(boxes[i][3] * w)

        roi = gray[ymin:ymax, xmin:xmax]
        if roi.size == 0: continue

        I = np.mean(roi) / 255.0
        S = ((xmax - xmin) * (ymax - ymin)) / (w * h)
        D = 1.0 / (S + 1e-6)
        
        P = 1.0 - abs((xmin + xmax)/2 - w/2) / (w/2)
        C = scores[i]

        G = (0.35*I + 0.25*S + 0.20*(1/D) + 0.10*P + 0.10*C) * 100
        vals.append(G)

    return float(np.clip(np.mean(vals), 0, 100)) if vals else 0.0

def detect_lanes(frame):
    return np.zeros_like(frame)
