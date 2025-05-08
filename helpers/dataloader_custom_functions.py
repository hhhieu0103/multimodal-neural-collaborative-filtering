import torch

from helpers.mem_map_dataloader import MemMapDataLoader


def collate_fn(batch):
    users = [x[0] for x in batch]
    items = [x[1] for x in batch]
    ratings = [x[2] for x in batch]
    feature_dicts = [x[3] for x in batch]
    images = [x[4] for x in batch]
    audio = [x[5] for x in batch]

    users = torch.stack(users)
    items = torch.stack(items)
    ratings = torch.stack(ratings)

    features = None
    sample_dict = feature_dicts[0]
    if sample_dict is not None:
        features = {}
        feature_names = list(sample_dict.keys())
        for name in feature_names:
            feature_tensors = [feature_dict[name] for feature_dict in feature_dicts]
            if feature_tensors[0].dtype == torch.float32:
                features[name] = torch.stack(feature_tensors)
            else:
                indices = torch.cat(tuple(feature_tensors))
                lengths = torch.tensor([len(indices) for indices in feature_tensors], dtype=torch.short)
                offsets = torch.zeros(lengths.size(0), dtype=torch.long)
                torch.cumsum(lengths[:-1], dim=0, out=offsets[1:])
                features[name] = (indices, offsets)

    sample_image = images[0]
    if sample_image is not None:
        images = torch.stack(images)
    else:
        images = None

    sample_audio = audio[0]
    if sample_audio is not None:
        audio = torch.stack(audio)
    else:
        audio = None

    return users, items, ratings, features, images, audio


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    if isinstance(dataset.image_dataloader, MemMapDataLoader):
        dataset.image_dataloader.open_lmdb()

    if isinstance(dataset.audio_dataloader, MemMapDataLoader):
        dataset.audio_dataloader.open_lmdb()