import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TeacherEmbeddingDataset(Dataset):
    def __init__(self, teacher_path: Path):
        """
        :param teacher_path: Path to a directory containing teacher [name].npy
        and a jsonl files with the inputs with the same name
        """

        # teacher files paths, get all the .npy files
        self.teacher_path = teacher_path

        pattern = re.compile(r"embeddings-(\d+)-(\d+).npy")
        embedding_files = list(teacher_path.rglob("*/embeddings*"))

        _embedding_files = []
        for file in embedding_files:
            match = pattern.search(str(file))
            if match:
                _embedding_files.append(
                    (int(match.group(1)), int(match.group(2)), file)
                )

        embedding_files = _embedding_files

        input_files = [
            (
                start,
                end,
                Path(
                    str(file).replace("embeddings-", "inputs-").replace("npy", "jsonl")
                ),
            )
            for start, end, file in embedding_files
        ]

        sample_counter = 0
        # load only text files
        self.texts = []
        self.inputidx2file_and_pos = []

        for (start, end, input_file), (_, _, embedding_file) in zip(
            input_files, embedding_files
        ):
            with open(input_file, "r") as f:
                inputs = [json.loads(line) for line in f]

            self.texts.extend(inputs)
            self.inputidx2file_and_pos.extend(
                [(i, embedding_file) for i in range(len(inputs))]
            )

            sample_counter += len(inputs)

        self.current_file_path = embedding_files[0][2]
        self.embeddings = np.load(self.current_file_path)
        #self.texts = [t.encode("utf-8").astype(np.string_) for t in self.texts]
        self.texts = np.array(self.texts)#
        self.inputidx2file_and_pos = pd.DataFrame(
            self.inputidx2file_and_pos, columns=["idx", "file"]
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[np.array, str]:
        try:
            needed_file = self.inputidx2file_and_pos.iloc[idx]["file"]
        except Exception as e:

            raise ValueError(f"Error with {idx}: {e}")
        if needed_file != self.current_file_path:
            self.current_file_path = needed_file
            del self.embeddings
            self.embeddings = np.load(self.current_file_path)

        loc_idx = int(self.inputidx2file_and_pos.iloc[idx]["idx"])
        return self.embeddings[loc_idx].astype(np.float32), self.texts[idx]

    def __repr__(self):
        return (
            f"{self.teacher_path} with {len(self)} samples"
            + "\n"
            + f"{self.embeddings.shape}, {len(self.texts)}"
        )


class TextDataset:

    def __init__(
        self,
        teachers_path,
    ):

        self.teacher_paths = teachers_path

        self.datasets = []
        for teacher in self.teacher_paths:
            ds = TeacherEmbeddingDataset(teacher)
            self.datasets.append(ds)

        self.texts2teachersidx = defaultdict(dict)

        for teacher_idx, dataset in enumerate(self.datasets):
            for idx, text in enumerate(dataset.texts):
                self.texts2teachersidx[text][teacher_idx] = idx
                
        self.texts = np.array(list(self.texts2teachersidx.keys()))#.astype(np.string_)
        print(len(self.texts))
        # make texts2teachersidx a dataframe indexed by the texts with the teacher indices as columns
        self.texts2teachersidx = pd.DataFrame(self.texts2teachersidx).T

    def roll(self, worker_id=-1, n_workers=-1):
        # rotate the texts so each worker has a different starting point
        if worker_id != -1 and n_workers != -1:
            N = len(self.texts)
            roll_size = N // n_workers
            shift = worker_id * roll_size
            self.texts = np.roll(self.texts, shift=shift)#.astype(np.string_)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[str, List[np.array], List[int]]:
        text = self.texts[idx]
        text_teachers_indices = self.texts2teachersidx.loc[text]

        aligned_embeddings = []
        teacher_indices = []

        for teacher_idx, loc_idx in text_teachers_indices.items():
            if pd.isnull(loc_idx):
                continue
            aligned_embeddings.append(self.datasets[int(teacher_idx)][int(loc_idx)][0])
            teacher_indices.append(teacher_idx)

        return text, aligned_embeddings, teacher_indices

    def __repr__(self):
        s = ""
        for i, dataset in enumerate(self.datasets):
            s += f"Teacher {i}: {dataset}\n"

        s += f"Aligned dataset with {len(self)} samples"

        return s


def make_aligned_collate_fn(tokenizer, teachers_dims: List[int]):
    def collate_fn(batch):
        """

        :param batch:
        :return: 3 lists of targets, inputs, teacher_idx. targets and inputs are lists of
        length n_teachers and contain the targets and inputs for each teacher.
        """

        # targets is List[List[np.array]], inputs is List[str], teacher_idx is List[int]
        inputs, aligned_embeddings, teacher_indices = zip(*batch)
        indexes = [i for i in range(len(inputs)) if inputs[i] is not None]

        mask = torch.zeros(len(teachers_dims), len(inputs), dtype=torch.bool)
        inputs = list(inputs)

        new_input = tokenizer(
            list(np.array(inputs)[indexes]),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        for (tokenizer_index, input_index) in zip(range(len(new_input["input_ids"])), indexes):#'token_type_ids' 'attention_mask'
            inputs[input_index] = {"input_ids": new_input["input_ids"][tokenizer_index], "attention_mask": new_input["attention_mask"][tokenizer_index], "token_type_ids": new_input["token_type_ids"][tokenizer_index]}


        targets = []
        for teacher in range(len(teachers_dims)):
            target = torch.zeros(
                len(indexes), teachers_dims[teacher], dtype=torch.float32
            )
            targets.append(target)

        local_index = 0
        for i, (_, aligned_embeddings, teacher_indices) in enumerate(batch):
            if i not in indexes:
                continue    
            for embedding, teacher_idx in zip(aligned_embeddings, teacher_indices):
                targets[teacher_idx][local_index] = torch.tensor(embedding)
                mask[teacher_idx, local_index] = True
            local_index += 1

        return inputs, targets, mask

    return collate_fn