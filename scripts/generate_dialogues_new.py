import os
import time
import re
import argparse
import json
from functools import wraps
from tqdm import tqdm
from runner import Runner


# =====================================================
# Global runners
# =====================================================

questioner_runner = None
oracle_runner = None


# =====================================================
# Setup runners
# =====================================================

def setup_runners(questioner_backend: str, oracle_backend: str):

    global questioner_runner, oracle_runner

    print(f"Initializing Questioner backend: {questioner_backend}")
    questioner_runner = Runner(questioner_backend)
    questioner_runner.set_model()

    print(f"Initializing Oracle backend: {oracle_backend}")
    oracle_runner = Runner(oracle_backend)
    oracle_runner.set_model()


# =====================================================
# Retry decorator
# =====================================================

def retry_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        while True:

            try:
                return func(*args, **kwargs)

            except Exception as e:

                print("Error:", e)
                print("Retrying in 5 seconds...")
                time.sleep(5)

    return wrapper


# =====================================================
# LLM call
# =====================================================

@retry_on_error
def llm_call(conversation, use_oracle=False):

    if use_oracle:

        output = oracle_runner.query_multi_turn(
            messages=conversation,
            max_tokens=32
        )

    else:

        output = questioner_runner.query_multi_turn(
            messages=conversation,
            max_tokens=512
        )

    return {
        "role": "assistant",
        "content": output.strip()
    }


# =====================================================
# Data processing
# =====================================================

def get_lists_of_candidates(contrast_sets):

    result = {}

    count = 0

    for contrast_set in contrast_sets.values():

        result[count] = {

            "candidates": contrast_set["items"],
            "target": contrast_set["target"]

        }

        count += 1

    return result


# =====================================================
# Prompt creation
# =====================================================

def get_prompts(candidates, target, stepwise=False):

    if stepwise:

        questioner = [

            {
                "role": "system",
                "content":
                "You are playing a guessing game.\n"
                "Ask as few yes/no questions as possible.\n\n"
                "Format output as:\n"
                "CANDIDATES: item, item\n"
                "QUESTION: question text"
            },

            {
                "role": "user",
                "content": f"This is the list of candidates: {candidates}"
            }

        ]

    else:

        questioner = [

            {
                "role": "system",
                "content":
                "You are playing a guessing game.\n"
                "Ask yes/no questions to identify the target."
            },

            {
                "role": "user",
                "content": f"This is the list of candidates: {candidates}"
            }

        ]

    oracle = [

        {
            "role": "system",
            "content":
            "You are the answerer in a guessing game.\n"
            "You must answer YES or NO.\n"
            "If guessed correctly, say: Yes! That's correct.\n\n"
            f"The target is: {target}"
        }

    ]

    return questioner, oracle


# =====================================================
# Dialogue generation
# =====================================================

@retry_on_error
def generate_dialogues(target_list_candidates, game_set, questioner_backend):

    output_dir = f"../data/generation/{questioner_backend}/{game_set}"

    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/dialogues.txt"

    stepwise = "stepwise" in game_set

    print("Saving to:", output_file)

    for index, value in tqdm(target_list_candidates.items()):

        successful = False

        while not successful:

            dialogue = []

            target = value["target"]

            print("\n====================")
            print("Target:", target)

            dialogue.append("====================")
            dialogue.append(f"target = {target}")

            questioner, oracle = get_prompts(
                ", ".join(value["candidates"]),
                target,
                stepwise
            )

            dialogue.append(f"candidates = {value['candidates']}")

            for turn in range(20):

                # Questioner asks

                q_out = llm_call(questioner, use_oracle=False)

                question_text = q_out["content"]

                print("Question:", question_text)

                dialogue.append("questioner: " + question_text)

                questioner.append({
                    "role": "assistant",
                    "content": question_text
                })

                try:

                    processed_question = question_text.split("QUESTION:")[1].strip()

                except:

                    processed_question = question_text

                oracle.append({
                    "role": "user",
                    "content": processed_question
                })

                # Oracle answers

                o_out = llm_call(oracle, use_oracle=True)

                answer = o_out["content"]

                print("Answer:", answer)

                dialogue.append("answerer: " + answer)

                questioner.append({
                    "role": "user",
                    "content": answer
                })

                oracle.append({
                    "role": "assistant",
                    "content": answer
                })

                # Stop condition

                if "correct" in answer.lower():

                    with open(output_file, "a") as f:

                        for line in dialogue:

                            f.write(line + "\n")

                    successful = True

                    break


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen",
        choices=["openai", "gemini", "qwen", "llama"]
    )

    parser.add_argument(
        "--oracle_backend",
        type=str,
        default="openai",
        choices=["openai", "gemini"]
    )

    parser.add_argument(
        "--game_set",
        type=str,
        default="8_mcrae"
    )

    args = parser.parse_args()

    print("Questioner:", args.backend)
    print("Oracle:", args.oracle_backend)

    # Setup runners

    setup_runners(
        questioner_backend=args.backend,
        oracle_backend=args.oracle_backend
    )

    # Load data

    game_set_clean = args.game_set.replace("_stepwise", "")

    with open(f"../data/game_sets/{game_set_clean}/contrast_sets.json") as f:

        contrast_sets = json.load(f)

    target_list_candidates = get_lists_of_candidates(contrast_sets)

    # Generate

    generate_dialogues(
        target_list_candidates,
        args.game_set,
        args.backend
    )
