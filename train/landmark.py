from typing import Dict

from .common import BaseTrainer

class LandmarkTrainer(BaseTrainer):
    """landmark regression model
    """
    def _fetch_input_label(self, samples: Dict):
        return samples["image"], samples["landmarks"]

if __name__ == "__main__":

    from torchvision.transforms import Compose
    from torch.optim.lr_scheduler import StepLR
    import torch

    from dataset import IBug300W
    from model.backbone import BackBoneFactory
    from model.landmark import BaselineNet
    from transform.landmark import RandomBlur, RandomCrop, \
        RandomHorizontalFlip, Resize, ToTensor, Normalize, \
        RandomRotate, RandomScale, Show
    from utils.dataset import load_ibug_300w

    from .metric import MetricAcc, MetricLoss, MetricManager


    datasets = load_ibug_300w()

    data_transforms = {
        "train": Compose([
            Resize((500, 500)),
            RandomBlur(),
            RandomHorizontalFlip(),
            RandomCrop((450, 450)),
            RandomRotate(),
            RandomScale(),
            Resize((224, 224)),
            ToTensor(),
            Normalize()
        ]),
        "val": Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize()
        ]),
    }
    for phase in ["train", "val"]:
        datasets[phase].transform = data_transforms[phase]

    # backbone_ = "resnet50"
    backbone_ = "resnet101"
    backbone = BackBoneFactory()(backbone_)
    net = BaselineNet(backbone)
    print(net)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    exp_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    metric_manager = MetricManager()
    metric_manager.add_stat("loss", MetricLoss())
    metric_manager.set_metric("loss")

    trainer = LandmarkTrainer(
        net,
        optimizer,
        criterion,
        scheduler=exp_lr_scheduler,
        datasets=datasets,
        metric_manager=metric_manager,
        model_name="landmark_baseline_{}".format(backbone_),
        num_epochs=100,
        batch_size=128,
        num_workers=8
    )

    net = trainer.train()

    epoch_stats = metric_manager.epoch_stat
    print(epoch_stats)