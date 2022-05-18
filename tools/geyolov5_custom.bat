@echo off

python geyolov5.py ^
--device 0 ^
--ckpt E:\tseng\models\yolov5s6_42.pt ^
--size 704 1280 ^
--graph-name person_car_yolov5m6.graph ^
--code-name person_car_yolov5m6 ^
--max-num-obj 10000