import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    lengths = {'spectrogram': 0, 'text_encoded': 0}
    result_batch = {'spectrogram': [], 'spectrogram_length': [],
                    'text_encoded': [], 'text_encoded_length': [],
                    'text': [], 'audio': [], 'audio_path': []}
    for key in lengths.keys():
        for i in range(len(dataset_items)):
            lengths[key] = max(lengths[key], dataset_items[i][key].shape[-1])
    for item in dataset_items:
        result_batch['spectrogram_length'].append(item['spectrogram'].shape[-1])
        result_batch['text_encoded_length'].append(item['text_encoded'].shape[-1])
        for key in lengths.keys():
            p = (0, lengths[key] - item[key].shape[-1])
            result_batch[key].append(F.pad(item[key], p, mode='constant', value=0))
        for key in ['text', 'audio', 'audio_path']:
            result_batch[key].append(item[key])
    for key in lengths.keys():
        result_batch[key] = torch.vstack(result_batch[key])
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    return result_batch
