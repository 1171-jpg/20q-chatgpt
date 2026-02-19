#!/bin/bash

mkdir -p logs

GAME_SET=8_mcrae

echo "Running Qwen on GPU 0"
CUDA_VISIBLE_DEVICES=0 python generate_dialogues_new.py \
    --backend qwen \
    --game_set $GAME_SET \
    > logs/qwen.log 2>&1 &


echo "Running Llama on GPU 1"
CUDA_VISIBLE_DEVICES=4 python generate_dialogues_new.py \
    --backend llama \
    --game_set $GAME_SET \
    > logs/llama.log 2>&1 &


echo "Running OpenAI (no GPU needed)"
python generate_dialogues_new.py \
    --backend openai \
    --game_set $GAME_SET \
    > logs/openai.log 2>&1 &

echo "Running Gemini (no GPU needed)"
python generate_dialogues_new.py \
    --backend gemini \
    --game_set $GAME_SET \
    > logs/gemini.log 2>&1 &

wait

echo "All jobs finished"
