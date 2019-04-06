from retinanet.dataloader.data_loaders import BoschDataset, collater
from retinanet.dataloader.transformers import AspectRatioBasedSampler
from torch.utils.data import DataLoader

def test_bosch_dataset():
    dataset = BoschDataset("data/bosch_sample.yaml")

    sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    for batch in dataloader:
        print(batch)

