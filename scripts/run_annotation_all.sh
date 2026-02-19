#!/bin/bash

# ==========================================
# Run annotation for multiple backends and game sets
# ==========================================

BACKENDS=("gemini" "openai","llama","qwen")
# GAME_SETS=("8_mcrae" "16_mcrae" "8_wordnet" "8_gpt" "8_mcrae_stepwise")
GAME_SETS=("8_mcrae")
echo "========================================="
echo "Starting oracle annotation batch run"
echo "========================================="

for BACKEND in "${BACKENDS[@]}"
do
    for GAME_SET in "${GAME_SETS[@]}"
    do

        echo ""
        echo "-----------------------------------------"
        echo "Backend: $BACKEND"
        echo "Game set: $GAME_SET"
        echo "-----------------------------------------"

        python generate_oracle_annotations_new.py \
            --backend "$BACKEND" \
            --game_set "$GAME_SET"

        echo "Saved to:"
        echo "../data/generation/$BACKEND/$GAME_SET/oracle_annotations.json"

    done
done

echo ""
echo "========================================="
echo "All annotation runs complete"
echo "========================================="
