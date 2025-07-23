# training - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 2

### Label Distribution

- training: 2 issues
- stale: 2 issues

---

## Issue #N/A: Segmentation Fault on GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/7337
**State**: closed
**Created**: 2024-05-17T09:17:40+00:00
**Closed**: 2024-08-09T01:07:02+00:00
**Comments**: 10
**Labels**: training, stale

### Description

When I am trying to run the following finetuning command on GPU:
**nohup ../build/bin/finetune --model-base llama-3b-Q5_0.gguf --train-data "shakespeare.txt" --save-every 1 --adam-iter 2 --batch 4 --ctx 4 --lora-out ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/lora.bin --checkpoint-in ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/checkpoint.gguf --checkpoint-out ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/checkpoint-ITERATION.gguf > ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/training_logs.out -ngl 33**

I get **segmentation fault** error with ever increasing **nohup.out** file:

llama_model_loader: loaded meta data with 24 key-value pairs and 237 tensors from llama-3b-Q5_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_mo

[... truncated for brevity ...]

---

## Issue #N/A: Training Custom using train-text-from-scratch

**Link**: https://github.com/ggml-org/llama.cpp/issues/2049
**State**: closed
**Created**: 2023-06-29T15:21:57+00:00
**Closed**: 2024-04-09T01:08:35+00:00
**Comments**: 3
**Labels**: training, stale

### Description

Hi.

I have been trying to train some custom data using the base model file: open-llama-7B-open-instruct.ggmlv3.q4_0.bin

Here is the command Im running

`./bin/train-text-from-scratch \
        --vocab-model ../models/ggml-vocab.bin \
        --ctx 64 --embd 256 --head 8 --layer 16 \
        --checkpoint-in  ik.bin \
        --checkpoint-out ik.bin \
        --model-out ik.bin \
        --train-data ik.txt \
        -t 6 -b 16 -n 32 --seed 1 --adam-iter 16 \
        --print-details-interval 0 --predict 16 \
        --mem-model 12 --mem-compute 12  --use-flash`


I'm able to train the data but I have following 2 concerns:

1. How Can I pass multiple text files instead of 1 text file? I have about 1 lakh+ text file which I need to train the model. Should I combine the text files?
2. I have a GPU of 24GB VRAM. But its not able to utilize more than 1gb and hence the process of training is slow. 

---

