import os
import re
import json
import time
import argparse
from functools import wraps
from tqdm import tqdm

from runner import Runner


# =====================================================
# Global oracle runner
# =====================================================

oracle_runner = None


# =====================================================
# Setup oracle runner
# =====================================================

def setup_oracle_runner(backend):

    global oracle_runner

    print(f"Initializing Oracle backend: {backend}")

    oracle_runner = Runner(backend)

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
# Dialogue reader
# =====================================================

def read_dialogues(path):

    with open(path) as f:

        dialogues = f.read().split("====================")

    dialogues = [

        dialogue.split("\n")[1:]

        for dialogue in dialogues

        if dialogue.strip() != ""

    ]

    return dialogues


# =====================================================
# Oracle call using Runner
# =====================================================

@retry_on_error
def oracle_call(target, question):

    conversation = [

        {
            "role": "system",
            "content":
            "You are the answerer in a guessing game.\n"
            "You must answer ONLY yes or no.\n"
            f"The target is: {target}"
        },

        {
            "role": "user",
            "content": question
        }

    ]

    output = oracle_runner.query_multi_turn(

        messages=conversation,
        max_tokens=32

    )

    text = output.strip().lower()

    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    else:
        return "no"


# =====================================================
# Annotation generator
# =====================================================

@retry_on_error
def get_complete_answers(dialogues, annotation_file_path):

    # Resume support

    if os.path.exists(annotation_file_path):

        with open(annotation_file_path) as f:

            annotations = json.load(f)

    else:

        annotations = []

        for i in range(len(dialogues)):

            annotations.append({

                "annotation_status": "ongoing",
                "target": "",
                "candidates": [],
                "dialogue_id": i + 1,
                "questions": []

            })


    # Find resume point

    annotation_point = 1

    for ann in annotations:

        if "annotation_status" in ann:

            annotation_point = ann["dialogue_id"]

            break


    # Process dialogues

    for dialogue_id, dialogue in enumerate(

        tqdm(dialogues[annotation_point - 1:]),

        start=annotation_point

    ):

        target = re.sub("target = ", "", dialogue[0])


        candidates = re.sub(

            "candidates = ",
            "",
            dialogue[1]

        ).strip().replace("[", "").replace("]", "").replace("'", "").split(", ")


        questions = [

            re.sub("questioner: ", "", line)

            for line in dialogue

            if "questioner:" in line

        ]


        answers_per_target = [

            re.sub("answerer: ", "", line)

            for line in dialogue

            if "answerer:" in line

        ]


        dialogue_dict = {

            "annotation_status": "ongoing",
            "target": target,
            "candidates": candidates,
            "dialogue_id": dialogue_id,
            "questions": annotations[dialogue_id - 1]["questions"]

        }


        start_q = len(dialogue_dict["questions"])


        for q_index in tqdm(range(start_q, len(questions))):

            question = questions[q_index]

            item_specific_answers = {}


            for candidate in candidates:

                if candidate == target:

                    oracle_output = answers_per_target[q_index].lower()

                else:

                    oracle_output = oracle_call(candidate, question)


                if "yes" in oracle_output:
                    item_specific_answers[candidate] = "yes"
                else:
                    item_specific_answers[candidate] = "no"


            question_dict = {

                "question_step": q_index + 1,
                "question": question,
                "item_specific_answers": item_specific_answers

            }


            dialogue_dict["questions"].append(question_dict)


            annotations[dialogue_id - 1].update(dialogue_dict)


            with open(annotation_file_path, "w") as f:

                json.dump(annotations, f, indent=4)


        del annotations[dialogue_id - 1]["annotation_status"]


        with open(annotation_file_path, "w") as f:

            json.dump(annotations, f, indent=4)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen"
    )

    parser.add_argument(
        "--oracle_backend",
        type=str,
        default="openai"
    )

    parser.add_argument(
        "--game_set",
        type=str,
        default="8_mcrae"
    )

    args = parser.parse_args()


    print("Questioner backend:", args.backend)
    print("Oracle backend:", args.oracle_backend)


    setup_oracle_runner(args.oracle_backend)


    data_dir = f"../data/generation/{args.backend}/{args.game_set}"


    dialogues_path = f"{data_dir}/dialogues.txt"
    annotation_path = f"{data_dir}/oracle_annotations.json"


    print("Reading:", dialogues_path)
    print("Saving:", annotation_path)


    dialogues = read_dialogues(dialogues_path)


    get_complete_answers(dialogues, annotation_path)
