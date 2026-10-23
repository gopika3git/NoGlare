# convert_fp16.py
import tensorflow as tf

pb_path = "inference_graph/frozen_inference_graph.pb"

input_arrays = ["image_tensor"]
output_arrays = ["detection_boxes", "detection_scores"]

print("[*] Initiating FP16 Converter Engine with Flex Ops...")

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=pb_path,
    input_arrays=input_arrays,
    output_arrays=output_arrays,
    input_shapes={"image_tensor": [1, 180, 320, 3]} 
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# =====================================================================
# FIX: ALLOW FLEX OPS FALLBACK BACKBONE KERNELS
# =====================================================================
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS   # Fallback on full TF ops (Flex Ops)
]
# =====================================================================

try:
    tflite_fp16_model = converter.convert()
    output_model_path = "inference_graph/glare_detector_fp16.tflite"
    with open(output_model_path, "wb") as f:
        f.write(tflite_fp16_model)
    print(f"[+] Success! Saved FP16 quantized model to: {output_model_path}")
except Exception as e:
    print(f"[!] Quantization failed: {str(e)}")
