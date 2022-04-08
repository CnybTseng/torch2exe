@echo off

python geyolox.py ^
--exp_file ..\..\YOLOX\exps\default\yolox_m.py ^
--cuda ^
--graph-name yolox_m.graph ^
--code-name yolox_m ^
--ckpt ..\..\..\models\YOLOX\yolox_m.pth