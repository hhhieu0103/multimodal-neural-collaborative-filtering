import torch

def collate_fn(batch):
    users = [x[0] for x in batch]
    items = [x[1] for x in batch]
    ratings = [x[2] for x in batch]
    feature_dicts = [x[3] for x in batch]
    images = [x[4] for x in batch]

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
                lengths = torch.tensor([len(indices) for indices in feature_tensors], dtype=torch.long)
                offsets = torch.zeros(lengths.size(0), dtype=torch.long)
                torch.cumsum(lengths[:-1], dim=0, out=offsets[1:])
                features[name] = (indices, offsets)

    sample_image = images[0]
    if sample_image is not None:
        images = torch.stack(images)
    else:
        images = None

    return users, items, ratings, features, images