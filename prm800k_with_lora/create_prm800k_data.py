import argparse
import glob
import os

import pandas as pd


def load_jsonl_to_dataframe(filepath):
    df = pd.read_json(path_or_buf=filepath, lines=True)
    phase, train_or_test = os.path.basename(filepath).split(".")[0].split("_")
    df["phase"] = phase
    df["type"] = train_or_test
    return df


def process_step(step_num, step, past_completions, past_ratings):
    data = []

    if step.get("completions", None) is None:
        return data

    for completion in step.get("completions", None):
        data.append(
            {
                "completion_step_num": step_num,
                "past_completions": past_completions,
                "past_ratings": past_ratings,
                "current_completion": completion["text"],
                "current_rating": completion["rating"],
            }
        )
    if step.get("human_completion", None) is not None:
        data.append(
            {
                "completion_step_num": step_num,
                "past_completions": past_completions,
                "past_ratings": past_ratings,
                "current_completion": step["human_completion"]["text"],
                "current_rating": 1.0,
            }
        )
    return data


def process_label(label):
    main_lvl1_keys = [
        "finish_reason",
        "total_time",
    ]
    data = {k: label.get(k, None) for k in main_lvl1_keys}

    unfolded_steps = []
    past_completions = []
    past_ratings = []

    for i, step in enumerate(label["steps"]):
        unfolded_steps += process_step(
            i, step, past_completions.copy(), past_ratings.copy()
        )

        if step["chosen_completion"] is not None:
            past_completions.append(
                step["completions"][step["chosen_completion"]]["text"]
            )
            past_ratings.append(
                step["completions"][step["chosen_completion"]]["rating"]
            )
        else:
            if step.get("human_completion", None) is not None:
                past_completions.append(step["human_completion"]["text"])
                past_ratings.append(1.0)
            else:
                past_completions.append(None)
                past_ratings.append(None)

    for unfolded_step in unfolded_steps:
        unfolded_step.update(data)

    return unfolded_steps


def process_question(question):
    main_lvl1_keys = [
        "problem",
        "ground_truth_answer",
        "ground_truth_answer",
        "pre_generated_steps",
        "pre_generated_answer",
        "pre_generated_verifier_score",
    ]
    data = {k: question.get(k, None) for k in main_lvl1_keys}
    return data


def process_row(row_indx, row):
    main_lvl1_keys = [
        "type",
        "phase",
        "labeler",
        "timestamp",
        "generation",
        "is_quality_control_question",
        "is_initial_screening_question",
    ]
    row = row.to_dict()
    data = {k: row.get(k, None) for k in main_lvl1_keys}
    data["data_id"] = row_indx

    # process question
    data.update(process_question(row["question"]))

    # process label
    responses = process_label(row["label"])
    for response in responses:
        response.update(data)

    return responses


def main(project_path):
    # Load data
    data_path = os.path.join(project_path, "prm800k", "data")
    file_paths = glob.glob(os.path.join(data_path, "phase2*.jsonl"))
    print(file_paths)
    df = pd.concat(
        [load_jsonl_to_dataframe(file_path) for file_path in file_paths],
        ignore_index=True,
    )

    # Process data
    processed_rows = (
        df.apply(lambda row: process_row(row.name, row), axis=1)
        .explode()
        .reset_index(drop=True)
    )
    processed_rows = processed_rows[~processed_rows.isnull()]
    processed_df = pd.DataFrame(list(processed_rows))

    # TODO - Think about this
    processed_df["past_completions"] = processed_df["past_completions"].apply(
        lambda pst_comp: "\n".join([f"Step {i}: {x}" for i, x in enumerate(pst_comp)])
    )

    final_df = processed_df[
        [
            "type",
            "phase",
            "data_id",
            "problem",
            "ground_truth_answer",
            "finish_reason",
            "completion_step_num",
            "current_completion",
            "current_rating",
            "past_completions",
            # "past_ratings",
            # "total_time",
            # "labeler",
            # "timestamp",
            # "generation",
            # "is_quality_control_question",
            # "is_initial_screening_question",
            "pre_generated_steps",
            "pre_generated_answer",
            # "pre_generated_verifier_score",
        ]
    ]

    train_df = final_df[final_df["type"] == "train"]
    test_df = final_df[final_df["type"] == "test"]

    train_df.to_json("prm800k_train_phase2_withgen.json", orient="records", lines=True)
    test_df.to_json("prm800k_test_phase2_withgen.json", orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PRM800k dataset")
    parser.add_argument(
        "--prm800k-path",
        type=str,
        help="Path to PRM800k dataset",
        default="/home/mila/m/maryam.hashemzadeh/scratch/verifier/prm800k",
    )
    args = parser.parse_args()
    main(args.prm800k_path)
