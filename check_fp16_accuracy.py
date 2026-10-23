# check_fp16_accuracy.py
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import json

if len(sys.argv) < 2:
    print("Error: Provide path input to testing video file asset (e.g., test_vedio4.mp4).")
    sys.exit(1)

video_path = sys.argv[1]
pb_path = "inference_graph/frozen_inference_graph.pb"
tflite_path = "inference_graph/glare_detector_fp16.tflite"
temp_cache = "inference_graph/pb_accuracy_cache.json"

if not os.path.exists(pb_path) or not os.path.exists(tflite_path):
    print("Error: Critical model files missing inside inference_graph/.")
    sys.exit(1)

# PASS 1: Extract Ground Truth safely
print("[*] PASS 1: Extracting Ground Truth from Original PB Model...")
pb_graph = tf.Graph()
with pb_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

pb_sess = tf.compat.v1.Session(graph=pb_graph)
pb_image_tensor = pb_graph.get_tensor_by_name('image_tensor:0')
pb_boxes_tensor = pb_graph.get_tensor_by_name('detection_boxes:0')
pb_scores_tensor = pb_graph.get_tensor_by_name('detection_scores:0')

cap = cv2.VideoCapture(video_path)
pb_cache = {}
frame_idx = 0

while frame_idx < 100:  # 100 frames is perfect for a safe, low-RAM evaluation sample
    ret, frame = cap.read()
    if not ret: break
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_NEAREST)
    inp = np.expand_dims(small, 0)
    pb_boxes, pb_scores = pb_sess.run([pb_boxes_tensor, pb_scores_tensor], feed_dict={pb_image_tensor: inp})
    pb_boxes, pb_scores = np.squeeze(pb_boxes), np.squeeze(pb_scores)
    
    valid_detections = []
    for idx in range(min(10, len(pb_scores))):
        if pb_scores[idx] >= 0.45:
            valid_detections.append({"box": pb_boxes[idx].tolist(), "score": float(pb_scores[idx])})
    pb_cache[str(frame_idx)] = valid_detections
    frame_idx += 1

cap.release()
pb_sess.close()

with open(temp_cache, "w") as f:
    json.dump(pb_cache, f)
del pb_graph, pb_cache
print("[+] Pass 1 Complete! Original Model closed and RAM completely flushed.")

# PASS 2: Evaluate FP16 Model accuracy
print("\n[*] PASS 2: Booting FP16 Interpreter Engine to Compare...")
with open(temp_cache, "r") as f:
    ground_truth = json.load(f)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
tflite_inputs = interpreter.get_input_details()
tflite_outputs = interpreter.get_output_details()

def calculate_iou(boxA, boxB):
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0])
    boxBArea = (boxB[3] - boxB[1]) * (boxB[2] - boxB[0])
    if float(boxAArea + boxBArea - interArea) <= 0: return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

total_iou, iou_counts, score_diffs = 0.0, 0, []
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while frame_idx < 100:
    ret, frame = cap.read()
    if not ret: break
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_NEAREST)
    inp = np.expand_dims(small, 0).astype(np.uint8)
    
    interpreter.set_tensor(tflite_inputs[0]['index'], inp)
    interpreter.invoke()
    tflite_boxes = np.squeeze(interpreter.get_tensor(tflite_outputs[0]['index']))
    tflite_scores = np.squeeze(interpreter.get_tensor(tflite_outputs[1]['index']))
    
    tflite_valid = [idx for idx in range(min(10, len(tflite_scores))) if tflite_scores[idx] >= 0.45]
    pb_valid = ground_truth.get(str(frame_idx), [])
    
    for pb_det in pb_valid:
        best_iou, matched_score = 0.0, 0.0
        p_box = pb_det["box"]
        for t_idx in tflite_valid:
            iou = calculate_iou(p_box, tflite_boxes[t_idx])
            if iou > best_iou:
                best_iou = iou
                matched_score = tflite_scores[t_idx]
        if best_iou > 0.1:
            total_iou += best_iou
            iou_counts += 1
            score_diffs.append(abs(pb_det["score"] - matched_score))
    frame_idx += 1

cap.release()
if os.path.exists(temp_cache): os.remove(temp_cache)

print("\n==================================================")
print("       FP16 ACCURACY CROSS-VALIDATION REPORT       ")
print("==================================================")
if iou_counts > 0:
    avg_iou = (total_iou / iou_counts) * 100
    avg_score_variance = np.mean(score_diffs) * 100
    print(f"[+] Spatial Bounding Box Match Accuracy (IoU): {avg_iou:.2f}%")
    print(f"[+] Prediction Confidence Deviation: {avg_score_variance:.2f}%")
    print(f"[+] Overall Model Accuracy Retention Score: {(avg_iou - avg_score_variance):.2f}%")
else:
    print("[!] No matching vehicles found in sample frames.")
print("==================================================")
