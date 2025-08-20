#!/usr/bin/env python3
"""
Tamasheq-French data processing.

Author
------
Marcely Zanon Boito 2022
"""

import json
import os
import torchaudio


def write_json(json_file_name, data):
    with open(json_file_name, mode="w", encoding="utf-8") as output_file:
        json.dump(
            data,
            output_file,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        )


def generate_json(folder_path, split):
    yaml_file = read_file(folder_path + "/" + split + ".yaml")
    translations_file = read_file(folder_path + "/" + split + ".hi")
    audio_paths = read_file(folder_path.replace("txt", "audio_paths"))

    assert len(yaml_file) == len(translations_file)

    output_json = dict()
    for i in range(len(yaml_file)):
        content = yaml_file[i]
        # utt_id = content.split(", wav: ")[1].split("}")[0]
        # utt_id = audio_paths[i].split("\t")[1].split("/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/data/train/segments/")[1].split(".wav")[0]
        utt_id = audio_paths[i].split(".wav")[0].split("/segments/")[1]
        duration = float(content.split("{duration: ")[1].split(",")[0])
        raw_path= audio_paths[i].split("\t")[1].replace("/raid/chandresh/Nivedita/STDATA/", "/Mamba-ASR/").replace("\n", "")
        # # print("raw_path:", raw_path)

        # try:
        #     waveform, sample_rate = torchaudio.load(raw_path)
        #     file_ok = True
        # except Exception as e:
        #     print(f"Failed to open audio file: {raw_path} ")
        #     file_ok = False

        # if duration >= 0.1 and file_ok:
        if duration >= 0.1 and duration <= 15:
            output_json[utt_id] = dict()
            # print("file:", raw_path)
            output_json[utt_id]["path"] = raw_path
            output_json[utt_id]["trans"] = translations_file[i]  # Assuming translations are line-separated 
            output_json[utt_id]["duration"] = content.split("{duration: ")[1].split(",")[0]                   
        # output_json[utt_id]["duration"] = content.split("{duration: ")[1].split(
        #     ","
        # )[0]

    return output_json


def read_file(f_path):
    return [line for line in open(f_path, encoding="utf-8")]


def data_proc(dataset_folder, output_folder):
    """

    Prepare .csv files for librimix

    Arguments:
    ----------
        dataset_folder (str) : path for the dataset github folder
        output_folder (str) : path where we save the json files.
    """

    try:
        os.mkdir(output_folder)
    except OSError:
        print(
            "Tried to create " + output_folder + ", but folder already exists."
        )

    for split in ["train", "valid", "test"]:
        split_folder = "/".join([dataset_folder, split, "txt"])

        output_json = generate_json(split_folder, split)

        write_json(output_folder + "/" + split + ".json", output_json)
