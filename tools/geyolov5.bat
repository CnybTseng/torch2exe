@echo off

python geyolov5.py ^
--device 0 ^
--ckpt ..\..\..\models\car-yolov5m6.pt ^
--size 704 1280 ^
--graph-name car_yolov5m6.graph ^
--code-name car_yolov5m6