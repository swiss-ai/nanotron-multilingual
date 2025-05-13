import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from nanotron import logging
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.logging import log_rank
from numba import jit

logger = logging.get_logger(__name__)


class MultilingualNanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        token_size (int): Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int = None,
        is_valid: bool = False,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
    ) -> None:

        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.is_valid = is_valid
        self.random_seed = random_seed
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            filename_pattern = "*.ds" if is_valid else "*0_unshuffled.ds" 
            try:
                self.datatrove_datasets.append(
                    DatatroveFolderDataset(
                        folder_path=dataset_folder,
                        filename_pattern=os.path.join(dataset_folder, filename_pattern),
                        seq_len=sequence_length,
                        recursive=False,
                        token_size=token_size,
                        shuffle=False,
                    )
                )
            except FileNotFoundError:
                warnings.warn(f"Dataset folder {dataset_folder} does not contain any .ds files. Skipping it.")
                self.datatrove_datasets.append([])

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        ## Set dataset weights
        if (dataset_weights is None):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            # Assign 0 weight to datasets with 0 samples
            self.dataset_weights = MultilingualNanoset.correct_weights_for_empty_datasets(np.array(dataset_weights), np.array(self.dataset_lengths))
 
        ## Check for empty datasets
        #assert not any(np.array(self.dataset_lengths) == 0), f"Dataset with 0 samples detected: {np.array(self.dataset_folders)[np.array(self.dataset_lengths) == 0]}. Please remove it from the dataset_folders list."
        ## Check for weights
        assert len(dataset_folders) == len(self.dataset_weights), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        
        ## Build dataset index and dataset sample index
        if is_valid:  # Valid MultilingualNanoset
            self.dataset_index, self.dataset_sample_index = build_valid_nanoset_index(self.dataset_lengths)
        else:  # Train MultilingualNanoset
            self.dataset_index, self.dataset_sample_index = self.build_train_nanoset_index()

        # Log Nanoset info
        logger.info(f"Dataset created!")

        self.print_nanoset_info()

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: The input ids wrapped in a dictionary
        """
        dataset = self.dataset_index[idx]
        dataset_sample = self.dataset_sample_index[idx]

        tokens = self.datatrove_datasets[dataset][dataset_sample]
        tokens["lang_code"] = torch.tensor(dataset, dtype=torch.long)

        return tokens

    def build_train_nanoset_index(self) -> np.ndarray:
        """
        Build train dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_train_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=self.dataset_weights, dataset_sizes=self.dataset_lengths
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate([dataset_sample_index for _ in range(num_epochs)])
        # Just keep the necessary samples
        dataset_index = dataset_index[: self.train_split_num_samples]
        dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]

        return dataset_index, dataset_sample_index

    def print_nanoset_info(self):

        log_rank(
            f"> [{'Validation' if self.is_valid else 'Training'} dataset] Total number of samples: {len(self)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        log_rank(
            f"> [{'Validation' if self.is_valid else 'Training'} dataset] Total number of tokens: {len(self) * self.sequence_length}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            log_rank(
                f">   Total number of {'validation' if self.is_valid else 'training'} samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

    @staticmethod
    def correct_weights_for_empty_datasets(dataset_weights: np.ndarray, dataset_lengths: np.ndarray) -> List[float]:
        """
        Corrects the weights for empty datasets

        Args:
            dataset_weights (np.ndarray): The dataset weights

        Returns:
            List[float]: The corrected dataset weights
        """
        dataset_weights[dataset_lengths == 0] = 0.0
        return normalize(dataset_weights.tolist())

@jit(nopython=True, cache=True)
def build_train_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights.
    For train and valid splits we split each dataset_folder in train (first part) and valid splits. We set the offsets to the train lengths
    for generating the valid split
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty((n_samples,), dtype="long")  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):

        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples

        # Set the error of empty datasets to -1 to avoid selecting them
        for i in range(len(dataset_sizes)):
            if dataset_sizes[i] == 0:
                errors[i] = -1

        # Select the dataset with the highest error
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index


@jit(nopython=True, cache=True)
def build_valid_nanoset_index(dataset_lengths: List[int]) -> np.ndarray:
    """
    Build valid dataset index and dataset sample index
    """
    dataset_index = []
    dataset_sample_index = []

    for i, length in enumerate(dataset_lengths):
        dataset_index.extend([i] * length)
        dataset_sample_index.extend(range(length))

    return np.array(dataset_index, dtype="uint"), np.array(dataset_sample_index, dtype="long")
