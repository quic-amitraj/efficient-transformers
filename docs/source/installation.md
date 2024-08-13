# Pre-requisites
System Requirements:
1. [Supported Linux OS](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/#operating-systems) - Ubuntu, RHEL and AWS Linux 
2. [Pre-requisites installed](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Pre-requisites/pre-requisites/)
3. [Cloud AI 100 Platform and Apps SDK installed](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/)  
4. [Multi-device support enabled for model sharding](https://github.com/quic/cloud-ai-sdk/tree/1.12/utils/multi-device)
 
* **Use bash terminal**
* **If using ZSH terminal then "device_group" should be in single quotes e.g.  "--device_group [0]"**

# Linux Installation 
There are two different way to install efficient-transformers.

## Using SDK

* Donwload Apps SDK: [Cloud AI 100 Platform and Apps SDK install](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/)  


```bash
# Install using Apps SDK

bash install.sh —enable-qeff

```
## Using GitHub Repository

```bash

# Create Python virtual env and activate it. (Required Python 3.8)

python3.8 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip

# Clone and Install the QEfficient Repo.
pip install git+https://github.com/quic/efficient-transformers

``` 