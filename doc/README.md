# 关于图像缩放

原图宽高为sw, sh
目标图宽高为dw, dh
原图缩放后未填充的宽高(round了的)为udw, udh, 填充后等于dw, dh

水平填充: pw = dw - udw
垂直填充: ph = dh - udh
填充量   = 0,   1, 2,   3, 4,   5, 6, ...
半填充量 = 0, 0.5, 1, 1.5, 2, 2.5, 3, ...

关于四周的填充量, 有种写法是:
pad_left = round(pad_half - 0.1)  = 0, 0, 1, 1, 2, 2, 3, ...
pad_right = round(pad_half + 0.1) = 0, 1, 1, 2, 2, 3, 3, ...
pad_left与pad_right之和刚好等于pad

pad_top和pad_bottom的计算方式类似, 奇数行列左右(上下)填充量相等, 偶数行列左(上)填充量比右(下)填充量少一个像素