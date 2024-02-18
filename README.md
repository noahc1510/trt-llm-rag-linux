# 🚀 RAG on Linux using TensorRT-LLM and LlamaIndex 🦙

###  Hardware Requirement
- Chat with RTX is currently built for RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory.
- At least 100 GB of available hard disk space
- Tested on Ubuntu 22.04
- Latest NVIDIA GPU drivers

### System Requirement
- Nvidia Driver: `sudo apt install nvidia-driver-535`
- CUDA: `sudo apt install nvidia-cuda-toolkit`
- NCCL: 
  ```shell
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
  sudo dpkg -i cuda-keyring_1.0-1_all.deb
  sudo apt-get update
  sudo apt install libnccl2
  ```
- libmpi: `sudo apt install libopenmpi-dev`

### Installation
1. Install miniconda, create new environment and install pytorch=2.1.0, mpi4py=3.1.5, tensorrt-llm
   ```shell
   conda create -n trtllm python=3.10
   conda activate trtllm
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   conda install -c conda-forge mpi4py mpich
   pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-llm
    ``` 
   In China, you can use these command below without vpn:
   ```shell
   conda create -n trtllm python=3.10
   conda activate trtllm
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   conda install -c conda-forge mpi4py mpich
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.nvidia.com tensorrt-llm
    ```
2. Install the requirements
   ```shell
   pip install -r requirements.txt
    ```
3. Download models and build the engine
    - **Download tokenizer**: Download config.json, tokenizer.json, tokenizer.model, tokenizer_config.json from
 [Llama 2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) or [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1),
place it in `./model/llama/llama13_hf` or `./model/mistral/mistral7b_hf`
    - **Get Quantized weights**: Downlaod the LLaMa 2 13B AWQ 4bit and mistral 7B int4 quantized model weights form NGC:
llama_tp1.json, llama_tp1_rank0.npz from [Llama 13b int4](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.3) or
mistral_tp1.json, mistral_tp1_rank0.npz from [Mistral 7B int4](https://catalog.ngc.nvidia.com/orgs/nvidia/models/mistral-7b-int4-chat),
place it in `./model/llama/llama13_int4_awq_weights` or `./model/mistral/mistral7b_int4_quant_weights`
    - **Build the engine**: Run `build-llama.sh` or `build-mistral.sh`
   
    Make sure your directory will be built like this:
   ```
    model
        - llama
            - llama13_hf
                - config.json
                - tokenizer.json
                - tokenizer.model
                - tokenizer_config.json
            - llama13_int4_awq_weights
                - llama_tp1.json
                - llama_tp1_rank0.npz
            - llama13_int4_engine
                - config.json
                - llama_float16_tp1_rank0.engine
                - model.cache
        - mistral
    ```
4. Run the app
   ```shell
   python app.py
    ```


---
Original readme below:

forked from the [Official Installer](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/)
and [trt-llm-rag-windows](https://github.com/NVIDIA/trt-llm-rag-windows) 

<p align="center">
<img src="https://gitlab-master.nvidia.com/winai/trt-llm-rag-windows/-/raw/main/media/rag-demo.gif"  align="center">
</p>

Chat with RTX is a demo app that lets you personalize a GPT large language model (LLM) connected to your own content—docs, notes, videos, or other data. Leveraging retrieval-augmented generation (RAG), TensorRT-LLM, and RTX acceleration, you can query a custom chatbot to quickly get contextually relevant answers. And because it all runs locally on your Windows RTX PC or workstation, you’ll get fast and secure results.
Chat with RTX supports various file formats, including text, pdf, doc/docx, and xml. Simply point the application at the folder containing your files and it'll load them into the library in a matter of seconds. Additionally, you can provide the url of a YouTube playlist and the app will load the transcriptions of the videos in the playlist, enabling you to query the content they cover


The pipeline incorporates the LLaMa 2 13B model, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. For demonstration, the dataset consists of recent articles sourced from [NVIDIA Gefore News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) seeks to enhance prediction accuracy by leveraging an external datastore during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

## Getting Started

### Hardware requirement
- Chat with RTX is currently built for RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory.
- At least 100 GB of available hard disk space
- Windows 10/11
- Latest NVIDIA GPU drivers

<h3 id="setup"> Setup Steps </h3>
Ensure you have the pre-requisites in place:

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) 0.7v for Windows using the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows)

Command:
```
pip install tensorrt_llm==0.7 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
```
Prerequisites 
- [Python 3.10](https://www.python.org/downloads/windows/)
- [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Windows&target_arch=x86_64)
- [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
- [cuDNN](https://developer.nvidia.com/cudnn)

More details in trt-llm page

2. Install requirement.txt
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121

pip install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir

pip uninstall -y nvidia-cudnn-cu11
```

3. In this project, the LLaMa 2 13B AWQ 4bit and mistral 7B int4 quantized model is employed for inference. Before using it, you'll need to compile a TensorRT Engine specific to your GPU for both the models. Below are the step to build the engine

- **Download tokenizer:** Ensure you have access to the [Llama 2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) repository on Huggingface.Downlaod config.json, tokenizer.json, tokenizer.model, tokenizer_config.json for both the models. Place the tokenizer files in dir <model_tokenizer>

- **Get Quantized weights:** Downlaod the LLaMa 2 13B AWQ 4bit and mistral 7B int4 quantized model weights form NGC:

    [Llama 13b int4](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.3), [Mistral 7B int4](https://catalog.ngc.nvidia.com/orgs/nvidia/models/mistral-7b-int4-chat)

- **Get TRT-LLM exmaple repo**: Download [TRT-LLM 0.7v](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.7.0) repo to build the engine

- **Build TRT engine:** 
Commands to build the engines 

Llama 13B int4:
```
python TensorRT-LLM-0.7.0\examples\llama\build.py --model_dir <model_tokenizer_dir_path> --quant_ckpt_path <quantized_weights_file_path> --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --output_dir <engine_output_dir> --world_size 1 --tp_size 1 --parallel_build --max_input_len 3900 --max_batch_size 1 --max_output_len 1024
```

Mistral 7B int4:
```
python.exe TensorRT-LLM-0.7.0\examples\llama\build.py --model_dir <model_tokenizer_dir_path>  --quant_ckpt_path <quantized_weights_file_path> --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --output_dir <engine_output_dir> --world_size 1 --tp_size 1 --parallel_build --max_input_len 7168 --max_batch_size 1 --max_output_len 1024
```

- **Run app**
```
python app.py --trt_engine_path <TRT Engine folder> --trt_engine_name <TRT Engine file>.engine --tokenizer_dir_path <tokernizer folder> --data_dir <Data folder>

```
- **Run app**
Update the **config/config.json** with below details for both the models


| Name | Details |
| ------ | ------ |
| --model_path | Trt engine direcotry path |
| --engine | Trt engine file name |
| --tokenizer_path | Huggingface tokenizer direcotry |
| --trt_engine_path | Directory of TensorRT engine |
| --installed <> | Ture/False if model is installed or not |

**Command:**
```
python app.py
```

## Adding your own data
- This app loads data from the dataset/ directory into the vector store. To add support for your own data, replace the files in the dataset/ directory with your own data. By default, the script uses llamaindex's SimpleDirectoryLoader which supports text files in several platforms such as .txt, PDF, and so on.


This project requires additional third-party open source software projects as specified in the documentation. Review the license terms of these open source projects before use.
