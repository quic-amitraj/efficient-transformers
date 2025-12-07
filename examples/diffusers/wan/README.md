# WAN Lightning

### Install Diffusers Dependencies

```
pip install "QEfficient[diffusers]"
```
## Instructions on Subfunction and Blocking

**NOTE**:

1. To use **Subfunction**, we need to install **APPS Assert SDK**. The following compiler environment variable must be set:
```
QAIC_COMPILER_OPTS_UNSUPPORTED="-loader-inline-all=0"
```

2. To use **Blocking**, pass the mode, desired number of blocks as shown below:
```
ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=21 num_q_blocks=2
```
### Recommended command for 480P with subfunction and blocking
```
QAIC_COMPILER_OPTS_UNSUPPORTED="-loader-inline-all=0" ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=21 num_q_blocks=2 python3 examples/diffusers/wan/wan_lightning.py
```

## Height, width for different resolutions
| Resol. | Height | Width |
|--------|--------|-------|
| 180P   | 192    | 320   |
| 240P   | 240    | 432   |
| 320P   | 368    | 640   |
| 480P   | 480    | 832   |
| 720P   | 720    | 1280  |