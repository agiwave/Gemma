# Gemma

A simple gemma demo. This repo is just for learning and backup. The codes in this Gemma is only < 400 lines. I just run it on laptop CPU(no CUDA).

If we limit the block_size to smaller value( even 256 ), it still work well. 

Please forget all aka code here. It's a sample proxy to torch:

    aka.nn --> torch.nn
    aka.numpy --> torch + torch.nn.F

## Requirements

    python
    torch
    torchvision
    sentencepiece

## Prepare

Download gemma files from: https://www.kaggle.com/models/google/gemma

to folder like:

    data/tokenizer.model
    data/gemma-2b-it.ckpt

## Run

> python Gemma.py
