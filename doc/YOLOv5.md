# About graph exporting
we should modify yolov5\models\yolo.py->Detect:forward().

# About output decoding

bx = (2 * sigma(tx) - 0.5 + cx) * stride
by = (2 * sigma(ty) - 0.5 + cy) * stride
bw = pw * (2 * sigma(tw))^2
bh = ph * (2 * sigma(th))^2

pw = anchor_w * stride
ph = anchor_h * stride