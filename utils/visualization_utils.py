import cv2
import numpy as np

def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.5):

    im_height, im_width = image.shape[:2]

    for i in range(min(len(boxes), len(scores))):

        if scores[i] >= min_score_thresh:

            ymin, xmin, ymax, xmax = boxes[i]

            if use_normalized_coordinates:
                left = int(xmin * im_width)
                right = int(xmax * im_width)
                top = int(ymin * im_height)
                bottom = int(ymax * im_height)

            else:
                left, right, top, bottom = xmin, xmax, ymin, ymax

            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                line_thickness
            )

            label = "Vehicle: {:.2f}".format(scores[i])

            cv2.putText(
                image,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
