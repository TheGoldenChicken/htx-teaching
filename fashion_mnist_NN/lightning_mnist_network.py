# So as you perhaps saw in the mnist_network.py file, there is a whole lot of 'boilerplate' that needs to be written
# whenever we wanna train a new neural network.... Boilerplate meaning "the same kinda stuff we need to write everytime"
# So often, we wanna avoid this... we can do this by using the package called "Lightning" (previously called "Pytorch Lightning")
# This gives the LightningModule class, which implements a whole lotta features for us by itself, meaning we don't need to do any of the following:
# - Make a train loop
# - Make a test loop
# - Make a validation loop
# - Implement logging of stats
# - and all kinds of other stuff
# This really saves a lot of time and allows you to focus on other, more important stuff...
# ... but most importantly, it forces you do a lot of things you'd otherwies forget
# Like actually logging your training stats (all hail reproduciability)

# We import a lot of functions from our regular MNIST file...
from mnist_network import ThatMnistNet, collate_fn, get_and_process_data

# And we get Lightning
import lightning as L

# We still get 'regular' torch, as we use some functionality from that, such as their tensors
import torch

# From the creators who brought you lightning, prepare for torchmetrics...
from torchmetrics.functional import accuracy

class LightningMnistNet(L.LightningModule):
    def __init__(self, in_features=28*28, out_features=10, lr=0.001, scuffed_version=False):
        """
        Like the other one but waaaay cooler
        So basically, Lightning just implements a lot of functionality under the hood
        I reccommend using it when you've grasped what Torch does... overall

        Things you DON'T need to do with lightning models:
        1. Make sure data/model is both on cuda/cpu - it does this automatically
        2. Write a training loop - it does this automatically
        """

        super().__init__()
        self.model = ThatMnistNet(in_dim=28*28, out_dim=10)
        self.train_set, self.test_set = get_and_process_data()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        We really technically don't need a forward function
        Since we can just call self.model(input) in training_step and evaluate
        But we're lazy....
        """
        return self.model(x)
        
    def training_step(self, batch):
        """
        A training step in lightning only needs to calculate and return the loss
        Everything else is handled, under the hood
        """

        x, y = batch
        logits = self(x)
        loss = self.model.criterion(logits, y)

        # Honestly, I don't remember where the logs end up...
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        """
        Just a single evaluation step... so this is meant for finding accuracy, not getting updates
        """

        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.to(torch.long))
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    """
    I don't quite remember if both below are actually necessary
    But I made this a while ago...
    """

    def validation_step(self, batch):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test") 

    def configure_optimizers(self):
        """
        Necessary since Lightning does not inherit the optimizers of whatever model it uses, it needs its own
        """

        optimizer = torch.optim.Adam(
            self.model.parameters(),
        )
        return {"optimizer": optimizer}


    """
    All these below are necessary unless you really wanna specify your own data loaders for the trainer module in PyTorch?
    """
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_set, shuffle=True, collate_fn=collate_fn, batch_size=16)
        return train_loader
    
    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_set, shuffle=True, collate_fn=collate_fn, batch_size=16)
        return test_loader
    

if __name__ == "__main__":
    model = LightningMnistNet()
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model)
    trainer.test()