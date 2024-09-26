import torch


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
    result_batch = {key: [] for key in dataset_items[0].keys()}
    for item in dataset_items:
        for key in item.keys():
            result_batch[key].append(item[key])
    for key in result_batch.keys():
        result_batch[key] = torch.hstack(result_batch[key])
    return result_batch
