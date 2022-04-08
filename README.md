# torch2exe
Convert PyTorch model to executable program.

# Build XXX
> cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release ..
or
> cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%\..\install ..

$ cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/../install ..

# Install YOLOV5

1. Create environment
> conda create -n yolov5 python=3.6
> conda activate yolov5

2. Install PyTorch
> pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

3. Install requirements
> cd %ROOT%
> git clone https://github.com/ultralytics/yolov5
> cd yolov5
> pip install -r requirements.txt

# Generate model engine

1. Copy `gen_wts.py` from tensorrt to yolov5

2. Generate wts file
> python gen_wts.py -w ..\..\models\yolov5m6.pt -o ..\..\models\yolov5m6.wts

3. Generate engine file
> cd XXX\build
> test-algorithm path\to\some\image

> test-algorithm ../config/yolov5.json test.jpg --save_path ./objs/ --loops 1

# Memory checking (for Linux only)

$ wget https://sourceware.org/pub/valgrind/valgrind-3.18.1.tar.bz2
$ tar jxvf valgrind-3.18.1.tar.bz2
$ cd valgrind-3.18.1
$ ./autogen.sh
$ ./configure --prefix=$(pwd)/../../toolkits/valgrind-3.18.1
$ make -j
$ make install
$ ~/toolkits/valgrind-3.18.1/bin/valgrind --tool=memcheck --leak-check=full --log-file=memcheck.log ./test-algorithm ../config/yolov5.json test.jpg --save_path ./objs/ --loops 1



ncnn2mem [ncnnproto] [ncnnbin] [idcpppath] [memcpppath]
ncnn2mem alexnet.param alexnet.bin alexnet.id.h alexnet.mem.h


#ifdef _WIN32
#include <windows.h>
#endif

HINSTANCE handle = LoadLibrary("yolox.dll");
	if (handle) {
		fprintf(stderr, "load dll success\n");
		float *weight = (float *)GetProcAddress(handle, "yolox_param_partly2_param");
		if (weight) {
			fprintf(stderr, "find yolox_param_partly2_param!\n");
		}
		FreeLibrary(handle);
	}
	return 0;

dumpbin /exports

# 导出YOLOV5的计算图配置文件
1. 修改yolov5\models\yolo.py文件中Detect类的forward方法, 只执行self.m层的运算
2. 修改yolov5\detect.py, 在推理部分插入代码导出计算图配置文件


# YOLOV5
## yolov5m6 (10000 loops)
1280x1280 9.903136ms
1280x704  6.615786ms speed up[33.19%]

## car-yolov5m6 (10000 loops)
1280x1280 9.115098ms
1280x704  6.102302ms speed up[33.05%]


echo "/home/tseng/cores/core-%e-%p-%t" > /proc/sys/kernel/core_pattern


NVIDIA GeForce RTX 2080 SUPER

GeForce RTX 3090
