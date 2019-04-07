from retinanet.dataloader.data_loaders import BoschDataset, collater
from retinanet.dataloader.transformers import AspectRatioBasedSampler
from torch.utils.data import DataLoader


from retinanet.train import main

def test_bosch_dataset():
    dataset = BoschDataset("data/bosch_sample.yaml")

    sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    for batch in dataloader:
        print(batch)


#def test_bosch_train():
#    --bosch_path "data/bosch-tld/bosch_sample.yaml"
#    main()


