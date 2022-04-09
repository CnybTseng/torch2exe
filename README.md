# torch2exe
Convert PyTorch model to executable program.

## 1 Build torch2exe

### 1.1 Dependencies

CUDA-11.1.1 <br>
cuDNN-8.1.1 <br>
TensorRT-7.2.3.4 <br>
OpenCV-4.5.3 <br>
jsoncpp-1.9.4 <br>

### 1.2 Build torch2exe from source code 

Firstly, clone the repository: <br>
> git clone https://github.com/CnybTseng/torch2exe.git <br>

On windows: <br>
> cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%\..\install .. <br>
> ninja && ninja install <br>

On linux: <br>
> cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/../install .. <br>
> ninja && ninja install <br>
