# torch2exe
Convert PyTorch model to executable program, not just a TensorRT engine.

## 1 Build torch2exe

### 1.1 Dependencies

CUDA-11.1.1 <br>
cuDNN-8.1.1 <br>
TensorRT-7.2.3.4 <br>
OpenCV-4.5.3 <br>
jsoncpp-1.9.4 <br>

### 1.2 Build torch2exe from source code 

Firstly, clone the repository: <br>
> `git clone https://github.com/CnybTseng/torch2exe.git` <br>

It is highly recommended to compile the project with ninja. You need to install [ninja](https://ninja-build.org/) first. On Windows system, you also need to install [microsoft visual c++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

On windows: <br>
> `cd torch2exe` <br>
> `mkdir build && cd build` <br>
> `cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%\..\install ..` <br>
> `ninja && ninja install` <br>

On linux: <br>
> `cd torch2exe` <br>
> `mkdir build && cd build` <br>
> `cmake -G"Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/../install ..` <br>
> `ninja && ninja install` <br>

## 2 Convert PyTorch model to C++ source code

### 2.1 Make a slight modification to your PyTorch model definition code

Usually, on the target platform, the output decoder of the model does not have ready-made operators and needs to be customized in the form of plug-ins. Therefore, the calculation logic of this part should not be included in the PyTorch style description file of the model. Take YOLOv5m6 as an example, output 4 tensors, and then parse these 4 tensors to get the detection result of the object, then the details of parsing these 4 tensors are not suitable for direct translation into the calculation graph, so the original code should be slightly modified , feed forward until the output of these 4 tensors.

### 2.2 Perform model feed forward and generate model code
