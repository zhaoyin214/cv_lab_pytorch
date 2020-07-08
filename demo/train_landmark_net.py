from torchvision.transforms import Compose
from torch.optim.lr_scheduler import StepLR
import cv2
import torch

from dataset import IBug300W
from model.backbone import BackBoneFactory
from model.landmark import BaselineNet
from test.landmark import LandmarkTester
from train.landmark import LandmarkTrainer
from train.metric import MetricAcc, MetricLoss, MetricManager
from transform.landmark import RandomBlur, RandomCrop, \
    RandomHorizontalFlip, Resize, ToTensor, Normalize, \
    RandomRotate, RandomScale, Show
from utils.dataset import load_ibug_300w

def train(nets, criterions, lr, num_epochs, batch_size):
    datasets = load_ibug_300w()
    data_transforms = {
        "train": Compose([
            Resize((500, 500)),
            RandomBlur(),
            # RandomHorizontalFlip(),
            RandomCrop((450, 450)),
            RandomRotate(),
            RandomScale(),
            Resize((224, 224)),
            # Show(),
            ToTensor(),
            Normalize()
        ]),
        "val": Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize()
        ]),
        "test": Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize()
        ]),
    }
    for phase in ["train", "val", "test"]:
        datasets[phase].transform = data_transforms[phase]
    metric_manager = MetricManager()
    metric_manager.add_stat("loss", MetricLoss())
    metric_manager.set_metric("loss")

    for ckey, criterion in criterions.items():
        for nkey, net in nets.items():
            # train
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            exp_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            trainer = LandmarkTrainer(
                net,
                optimizer,
                criterion,
                scheduler=exp_lr_scheduler,
                datasets=datasets,
                metric_manager=metric_manager,
                model_name="landmark_baseline_{}_criterion_{}".format(nkey, ckey),
                num_epochs=num_epochs,
                batch_size=batch_size,
                num_workers=4
            )
            net = trainer.train()
            trainer.test()
            logs = metric_manager.logs
            print(logs)
            # test
            tester = LandmarkTester(net, datasets["test"])
            for cnt, (image, _, _) in enumerate(tester.batch_pred()):
                cv2.imshow("grid", image)
                cv2.waitKey(2000)
                cv2.imwrite(
                    "./output/demo/landmark_baseline_{}_criterion_{}_{}".format(
                        nkey, ckey, cnt
                    ),
                    image
                )
                if cnt > 5:
                    break
            cv2.destroyAllWindows()

    return None

if __name__ == "__main__":

    net_list = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    nets = {
        backbone: BaselineNet(BackBoneFactory()(backbone))
        for backbone in net_list
    }
    criterions = {
        "mse": torch.nn.MSELoss(),
        "mae": torch.nn.L1Loss(),
        "smooth_l1": torch.nn.SmoothL1Loss(),
    }
    lr = 0.0005
    num_epochs = 20
    batch_size = 128

    train(nets, criterions, lr, num_epochs, batch_size)