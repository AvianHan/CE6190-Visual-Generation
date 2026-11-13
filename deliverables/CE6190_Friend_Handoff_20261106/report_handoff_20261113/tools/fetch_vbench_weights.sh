#!/bin/bash
# Weights only (URLs copied from VBench fd18b3d vbench/utils.py); no code is fetched here.
C=/home/ecs-user/ce6190-runtime/a100x8-20261110/vbench_cache
set -x
dl(){ [ -s "$2" ] || curl -sfL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; ls -la "$2"; }
dl https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt $C/clip_model/ViT-B-32.pt &
dl https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt $C/clip_model/ViT-L-14.pt &
dl https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth $C/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth &
dl https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth $C/amt_model/amt-s.pth &
dl https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip $C/raft_model/models.zip &
dl https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth $C/dino_model/dino_vitbase16_pretrain.pth &
dl https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth $C/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth &
dl https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth $C/grit_model/grit_b_densecap_objectdet.pth &
dl https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth $C/caption_model/tag2text_swin_14m.pth &
dl https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth $C/ViCLIP/ViClip-InternVid-10M-FLT.pth &
wait
echo WEIGHTS_DONE
