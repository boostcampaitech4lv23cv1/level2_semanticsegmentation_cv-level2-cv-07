# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))