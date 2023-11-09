import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.datamodule import SampleDataModule
from src.modelmodule import SampleLitModule, MyParameters


def main():
    params = MyParameters()
    callbacks = []
    if params.early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=params.early_stopping_patience)
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./pl-checkpoints",
        filename="sample-{epoch:02d}-{val_loss:.03f}-{val_acc:.03f}",
        save_top_k=3,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    model = SampleLitModule()

    trainer = pl.Trainer(
        max_epochs=params.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model=model, datamodule=SampleDataModule(batch_size=params.batch_size))


if __name__ == "__main__":
    main()
