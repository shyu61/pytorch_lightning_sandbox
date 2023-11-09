import torch
from tqdm import tqdm
from src.datamodule import SampleDataModule
from src.modelmodule import SampleLitModule, MyParameters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = MyParameters()
    new_model = SampleLitModule.load_from_checkpoint(
        "./pl-checkpoints/sample-epoch=06-val_loss=0.348-val_acc=28.042.ckpt"
    )
    new_model.to(device)  # Move model to GPU if available

    datamodule = SampleDataModule(batch_size=params.batch_size)
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()

    val_loss, val_acc = 0, 0

    for X, y in tqdm(val_dataloader):
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            pred = new_model(X)
            val_loss += new_model.loss_fn(pred, y)
            val_acc += (pred.argmax(1) == y).sum().type(torch.float)

    num_batches = len(val_dataloader)
    val_loss /= num_batches

    size = len(val_dataloader.dataset)
    val_acc /= size

    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
