from typing import Callable, Mapping, Optional, Sequence

import gin
import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from datasets import Dataset
from latentis import PROJECT_ROOT
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.nn import ModuleDict
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from transformers import ViTForImageClassification
import wandb
from residual.data.dataset import get_dataset
from residual.nn.adapter import (
    Adapter,
)
from residual.nn.encoder import (
    Encoder,
)
from residual.nn.model_registry import get_vision_encoder
from residual.tracing import encoder_name2tracer
from residual.tracing.tracer import ResidualTracer


@gin.configurable
def get_residual_tracer(encoder_name: str, pooling_fn: Callable):
    encoder: Encoder = get_vision_encoder(name=encoder_name, pooling_fn=pooling_fn)

    tracer_cls: ResidualTracer = encoder_name2tracer[encoder_name]

    tracer = tracer_cls(
        module_name=encoder.name,
        encoder=encoder,
        raw=False,
        accumulate=False,
    )
    tracer.__enter__()
    return tracer


class LiTModule(pl.LightningModule):
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "optimizer"):
            if mode:
                self.optimizer.train()
            else:
                self.optimizer.eval()

    def eval(self):
        super().eval()
        if hasattr(self, "optimizer"):
            self.optimizer.eval()

    def __init__(
        self,
        name: str,
        encoder: "Encoder",
        classifier: nn.Module,
        build_optimizer: Callable[["LiTModule"], torch.optim.Optimizer],
        step2metrics: Mapping[str, MetricCollection],
        adapter: Optional[nn.Module] = None,
        encoder_train: bool = False,
        adapter_train: bool = True,
        classifier_train: bool = False,
    ) -> None:
        super().__init__()
        self.name = name

        self.encoder = encoder.requires_grad_(encoder_train)

        self.adapter = (adapter or nn.Identity()).requires_grad_(adapter_train)

        self.classifier = classifier.requires_grad_(classifier_train)

        self.step2metrics = ModuleDict(
            {f"{k}_metrics": v for k, v in step2metrics.items()}
        )

        self.encoder_train = encoder_train
        self.adapter_train = adapter_train
        self.classifier_train = classifier_train

        self.build_optimizer = build_optimizer
        self.autotuned_lr = None

    def encode(self, x, forward=False):
        x = self.encoder(x)

        if isinstance(self.adapter, Adapter) and not forward:
            return self.adapter.encode(x)

        return self.adapter(x)

    def forward(self, x, stage: str = "test"):
        x = self.encode(x, forward=True)
        # print(
        # f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        # )

        logits = self.classifier(x)

        return dict(logits=logits)

    def _step(self, batch, batch_idx, stage: str):
        if hasattr(self, "optimizer"):
            assert (
                stage != "train" or self.optimizer.param_groups[0]["train_mode"]
            ), "Optimizer is not in training mode."
        x = batch["x"]
        y = batch["y"]

        logits = self.forward(x, stage=stage)["logits"]

        metrics = self.step2metrics[f"{stage}_metrics"]

        metrics(preds=logits.argmax(dim=-1), target=y)
        self.log_dict(metrics, prog_bar=True)

        loss = F.cross_entropy(logits, y)
        self.log(f"{stage}/logits_loss", loss, prog_bar=True, batch_size=x.shape[0])

        if hasattr(self.adapter, "loss"):
            for k, v in self.adapter.loss().items():
                self.log(f"{stage}/{k}", v, prog_bar=True, batch_size=x.shape[0])
                loss = loss + v

        if stage == "train":
            if hasattr(self.adapter, "lambdas"):
                wandb.log(
                    {
                        "lambdas": wandb.Histogram(
                            self.adapter.lambdas.detach().cpu().numpy()
                        )
                    }
                )

        return dict(loss=loss)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        kwargs = {}
        if self.autotuned_lr is not None:
            kwargs["lr"] = self.autotuned_lr
        optimizer = self.build_optimizer(self, **kwargs)
        assert optimizer is not None, "Optimizer is None."

        self.optimizer = optimizer[
            "optimizer"
        ]  # schedule free optimizer needs .eval and .train
        return optimizer

    # def on_load_checkpoint(self, checkpoint):
    #     callback: ModelCheckpoint = self.trainer.checkpoint_callback
    #     save_path = Path(callback.dirpath)

    #     self.load_state_dict(
    #         torch.load(
    #             save_path / f"{self.name}.pt",
    #             map_location=self.trainer.strategy.root_device,
    #             weights_only=False,
    #         ).state_dict(),
    #     )

    # def on_save_checkpoint(self, checkpoint):
    #     callback: ModelCheckpoint = self.trainer.checkpoint_callback
    #     save_path = Path(callback.dirpath)

    #     torch.save(
    #         self,
    #         save_path / f"{self.name}.pt",
    #     )

    # def on_train_epoch_end(self):
    #     self.adapter.lambdas.data = F.normalize(self.adapter.lambdas, p=2, dim=0)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if not self.encoder_train:
            self.encoder.eval()

        if not self.adapter_train:
            self.adapter.eval()

        if not self.classifier_train:
            self.classifier.eval()

        self.optimizer.train()

    def on_train_epoch_end(self):
        self.optimizer.eval()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.optimizer.eval()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        if hasattr(self, "optimizer"):
            self.optimizer.eval()


@gin.configurable
def build_optimizer(model: "LiTModule", lr: float, **kwargs):
    optimizer = AdamWScheduleFree(model.parameters(), lr=lr, **kwargs)
    return dict(optimizer=optimizer)


@gin.configurable
class CentroidClassifier(nn.Module):
    @classmethod
    def from_tensor(cls, centroids: torch.Tensor):
        model = cls.__new__(cls)
        super(CentroidClassifier, model).__init__()
        model.register_buffer("centroids", centroids)
        return model

    def __init__(
        self,
        encoder_name: str,
        dataset_name: str,
    ):
        super().__init__()
        class_encodings = torch.load(
            PROJECT_ROOT / "classifiers" / dataset_name / f"{encoder_name}.pt",
            weights_only=True,
        )
        self.register_buffer("centroids", class_encodings["class_encodings"])

    @property
    def num_classes(self):
        return self.centroids.shape[1]

    def forward(self, x: torch.Tensor):
        centroids = self.centroids

        # x = F.normalize(x, p=2, dim=-1)

        return x @ centroids


@gin.configurable
class ViTClassifier(nn.Module):
    @classmethod
    def from_tensor(cls, centroids: torch.Tensor):
        model = cls.__new__(cls)
        super(ViTClassifier, model).__init__()
        model.register_buffer("centroids", centroids)
        return model

    def __init__(
        self,
        encoder_name: str,
    ):
        super().__init__()
        vit_model = ViTForImageClassification.from_pretrained(encoder_name)
        class_encodings = vit_model.classifier.weight.T
        self.register_buffer("centroids", class_encodings)

    @property
    def num_classes(self):
        return self.centroids.shape[1]

    def forward(self, x: torch.Tensor):
        centroids = self.centroids

        # x = F.normalize(x, p=2, dim=-1)

        return x @ centroids


@gin.configurable
class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, bias: bool):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=bias
        )

    def forward(self, x):
        return self.fc(x)


@gin.configurable
def build_linear_classifier(
    encoder: Encoder, dataset: str, bias: bool
) -> LinearClassifier:
    in_features = encoder.encoding_dim()
    num_classes = get_dataset(dataset, split="test").features["y"].num_classes

    return LinearClassifier(
        in_features=in_features,
        num_classes=num_classes,
        bias=bias,
    )


@gin.configurable
def get_logger(**kwargs):
    return WandbLogger(**kwargs)


@gin.configurable
def keep_filter(x, class_indices: Sequence[str]):
    return x in class_indices


@gin.configurable
def run(
    exp_type: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    tune_lr: bool,
    build_optimizer: Callable,
    trainer_args: Mapping[str, object],
    logger: pl.pytorch.loggers.Logger,
    encoder: Encoder,
    classifier: torch.nn.Module,
    encoder_train: bool,
    adapter_train: bool,
    classifier_train: bool,
    device: torch.device,
    adapter: Optional[torch.nn.Module] = None,
    ckpt_dir_name: str = "checkpoints",
):
    print(gin.operative_config_str())
    run_dir = PROJECT_ROOT / ckpt_dir_name / dataset_name / encoder.name
    if (len(list(run_dir.glob(f"{exp_type}.*")))) > 0:
        print(
            f"Skipping {ckpt_dir_name}>{exp_type}>{dataset_name}>{encoder.name} as it already exists."
        )
        if isinstance(encoder, ResidualTracer):
            encoder.__exit__(None, None, None)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    config_str = gin.operative_config_str()
    if logger is not None:
        artifact = wandb.Artifact(
            "gin-config", type="config", description="Hydra config file"
        )
        with artifact.new_file("config.gin") as f:
            f.write(config_str)
        logger.experiment.log_artifact(artifact)

        logger.experiment.config.update(
            {
                "exp_type": exp_type,
                "dataset_name": dataset_name,
                "encoder_name": encoder.name,
            }
        )
        if hasattr(encoder, "properties"):
            logger.experiment.config.update(
                {f"encoder.{k}": v for k, v in encoder.properties().items()}
            )
        if adapter is not None and hasattr(adapter, "properties"):
            logger.experiment.config.update(
                {f"adapter.{k}": v for k, v in adapter.properties().items()}
            )

    device = torch.device(device)

    train_dataset: Dataset = get_dataset(dataset_name, split="train")
    val_dataset: Dataset = get_dataset(dataset_name, split="val")
    test_dataset: Dataset = get_dataset(dataset_name, split="test")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=encoder.collate_fn,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        collate_fn=encoder.collate_fn,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=encoder.collate_fn,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
    )

    metrics = lambda prefix, num_classes: MetricCollection(  # noqa: E731
        {
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ),
        },
        prefix=prefix,
    )

    num_classes = test_dataset.features["y"].num_classes

    lit_model = LiTModule(
        name=exp_type,
        encoder=encoder,
        adapter=adapter,
        classifier=classifier,
        step2metrics={
            "train": metrics(prefix="train/", num_classes=num_classes),
            "val": metrics(prefix="val/", num_classes=num_classes),
            "test": metrics(prefix="test/", num_classes=num_classes),
        },
        build_optimizer=build_optimizer,
        encoder_train=encoder_train,
        adapter_train=adapter_train,
        classifier_train=classifier_train,
    )

    trainer_args = dict(trainer_args)

    callbacks: list = trainer_args.pop("callbacks", [])

    trainer = pl.Trainer(
        **trainer_args,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val/accuracy", mode="max", patience=5, verbose=True),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
            ModelCheckpoint(
                monitor="val/accuracy",
                mode="max",
                save_top_k=1,
                dirpath=run_dir,
                filename=exp_type,
                verbose=True,
            ),
            # DeviceStatsMonitor(),
            *callbacks,  # Ensure any additional callbacks from trainer_args are included
        ],
    )

    train = any((encoder_train, adapter_train, classifier_train))
    if train:
        assert exp_type != "base", "base exp_type is only for testing."
        if tune_lr:
            from lightning.pytorch.tuner import Tuner

            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                lit_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                min_lr=1e-3,
                num_training=50,
                attr_name="autotuned_lr",
            )
            lr = lr_finder.suggestion()
            trainer.optimizers[0].param_groups[0]["lr"] = lr

            print(f"Suggested learning rate: {lr}")

        trainer.fit(
            lit_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    trainer.test(
        lit_model, dataloaders=test_dataloader, ckpt_path="best" if train else None
    )

    logger.experiment.finish()
    wandb.finish()
    if isinstance(encoder, ResidualTracer):
        encoder.__exit__(None, None, None)
