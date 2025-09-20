# Accelerate Activation for Large Model

In this section, we will focus on run activattion acceleration design flow in HLS.

By migrating the computation of activation functions from traditional x86 servers to programmable logic (PL) and optimizing it, we aim to significantly improve computational speed.

## Installation and Tutorials

Follow the [Linux tcl flow tutorial for activation accelerator](https://github.com/ZHAODONG-LYU/activation_accelerator_tutorial/blob/main/tutorial/tutorial_activation_accelerator_linux.md)

* Generate HLS IP Core
* Generate bitstream
* PYNQ environment setup
* Verification with PYNQ

With this tutorial, you can run the baseline version, which is function implement without any optimization.

A complete test vector guide can be found at this path [`tutorial/testvector.md`](https://github.com/ZHAODONG-LYU/activation_accelerator_tutorial/blob/main/tutorial/testvector.md) or [`tutorial/测试向量.docx`](https://github.com/ZHAODONG-LYU/activation_accelerator_tutorial/blob/main/tutorial/%E6%B5%8B%E8%AF%95%E5%90%91%E9%87%8F.docx), examples can be found in [`prj/testvector_example`](https://github.com/ZHAODONG-LYU/activation_accelerator_tutorial/blob/main/prj/testvector_example).

