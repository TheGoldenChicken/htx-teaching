# Ok, so this is going to be an implementation of a feedforward neural network (FFnn)
# It is the "simplest" version of an NN you can make, all its layers are "just" the linear ones with ReLU activation function
# Usually, you'd split a lot of parts up, so you'd have one script for downloading data, one for defining the model, one for calling and training the mode, etc.
# I'm just doing everything in this one script for simplicity, but if you want, you can split it up into individual modules
# That said, most tutorials on PyTorch are absolutely fucking terrible crimes against mankind
# So this is really the lesser of two evils I think

import torch

# TQDM is usually used to display a kind of progress bar during training
from tqdm import tqdm

# Matplotlib is just for plotting stats and such, we don't do much of that in this specific script, but nice to have
import matplotlib.pyplot as plt

# Torchvision is a computervision oriented library, we only use it to get the dataset (fashion_mnist) and to transform the images therein
from torchvision import datasets, transforms

# torch.cuda.is_avaliable is good to use to check if you can actually use GPU... otherwise we just use the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"YAASSSS, I'm using {device}!!!")

def get_and_process_data():
    """
    Returns MNIST dataset
    Not really warrented to use an actual 'create data' function, but it is what you usually use...
    """
    # Since the MNIST dataset is loaded as PIL (pillow) images (stupid), we add transform=ToTensor() to convert them to numpy tensors
    # Download only downloads if it doesn't already exist
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    return train_set, test_set

# Collate function is automatically called whenever we get data from out torch dataloader
def collate_fn(batch):
    """
    Uses default collate (makes tuple of (inputs, labels))
    Then flattens inputs (from shape (28,28) to (784,) )
    """
    batch = torch.utils.data.default_collate(batch)
    batch[0] = batch[0].flatten(start_dim=1).to(device)
    batch[1] = batch[1].to(device)
    return batch

class ThatMnistNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, lr=0.001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_dim, out_features=64),
            torch.nn.Linear(in_features=64, out_features=128),
            torch.nn.Linear(in_features=128, out_features=32),
            torch.nn.Linear(in_features=32, out_features=out_dim)

            # Normally, we would include a softmax here after the final linear layer, however,
            # Torch.nn.CrossEntropyLoss (our loss function) automatically puts a softmax before, however
            ).to(device)

        # Optimizer and criterion (loss function) respectively
        # Adam is kinda meta, which is why we use it
        # CrossEntropy is just what is used for classification problems
        # Don't try to understand the theory behind crossentropy, just know it interprets outputs as 
        # Probabilities (normally the network just outputs numbers), CE makes the sum of all these numbers be 1
        # Meaning it looks kinda like probabilities (since they sum to 1)
        # The loss is then larger if the probabilities are distributed other places than they should be
        # For example, if the probs look like [0.5,0.5,0,0,0,0,0,0,0,0] and the right answer is 5, the loss is high since no probability mass is there
        # Also there's a huge amount of theory on what the term 'Cross Entropy' actually means, just ask me if you wanna know it
        self.optim = torch.optim.Adam(self.layers.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor):
        """
        One forward pass through the network
        Note: Because of some PyTorch Dundermethods (magic methods), present in the torch.nn.Module class
        This function is automatically called when we go self() or model(), since torch.nn.Module class implements the __call__ function as 
        def __call__(input):
            return self.forward(input)
        something like that anyways...
        """

        return self.layers(x)

    def test(self, test_data_loader):
        """
        For getting accuracy of the model depending on test
        Technically, you shuoldn't test too much on your test set... otherwise you might end up hyperparameter fitting to your test set
        and that is actually overfitting (just humans doing it instead)
        Instead, you SHOULD have a third 'validation set' for doing hyperparameter tuning
        But, we're lazy
        """
        
        total_acc = 0

        for input_batch, label_batch in test_data_loader:
            # Get predictions
            outs = self(input_batch)

            # Remember, outs are probabilities (so there's 10 for each input)
            # The classification the network wants to assign, must therefore be the probability with the larget value
            # We find that using argmax (dim=1, because dim=0 would be across batch dimension)
            classifications = torch.argmax(outs, dim=1)
            
            total_acc += (classifications == label_batch).sum().item()

        total_acc = total_acc / len(test_data_loader.dataset)

        return total_acc


# Ok so why do we define two different classes - One with a train function and one without?
# It's complicated, but it makes the next few exercises where you use lightning actually work... don't worry too much about it
# Now honstly, I could probably have come up with a better way of handling this... but I'm a little bit lazy
class ThatMnistNetThatCanTrain(ThatMnistNet):
    def __init__(self, in_dim, out_dim, lr=0.001, *args, **kwargs) -> None:
        super().__init__(in_dim, out_dim, lr=0.001, *args, **kwargs)

    def train(self, train_data_loader, epochs, test_data_loader=None):
        """
        Train the whole thing
        Remember, each epoch is a runthrough of our entire data
        We technically use Stochastic gradient descent since we use batches as estimators for our true gradient
        To get the true gradient, we'd technically have to go through all data for a single update!!!
        But batching just makes sense ya'know
        """
        
        epoch_losses = []
        # I think you only really need one epoch to converge since the problem is basically trivial
        for epoch in range(epochs):
            epoch_loss = 0

            # there is a better way of using tqdm instead of printing like 10 lines (one for each epoch)
            for input_batch, label_batch in tqdm(train_data_loader):

                # Calculate our outputs...
                outs = self(input_batch)

                # Get our loss between the output ("probabilities") and the true labels
                loss = self.criterion(outs, label_batch)

                # Use backpropagation to our gradient
                loss.backward()

                # Make our optimizer take a single step depending on the gradient
                self.optim.step()

                # IMPORTANT: ZERO THE GRADIENTS!!
                # Because of the way loss.backward() assigns gradients to each parameter in the network, if we didn't zero the gradients her
                # We'd keep getting larger and larger updates to our weights, and they'd diverge to hell and beyond
                # Remember this stupid step!!!!
                self.optim.zero_grad()

                # Log the epoch loss so we can check how it changes...
                epoch_loss += loss.detach().item()

            epoch_losses.append(epoch_loss)
            print(epoch_loss)

            if test_data_loader is not None:
                print(f"Current epoch: {epoch}, acc: ", self.test(test_data_loader))

    def plot_predictions(self, test_loader, num_predictions=12):
        
        xs, targets = next(iter(test_loader))

        preds = torch.argmax(self(xs), dim=1)

        fig, axs = plt.subplots(4, 3, figsize=(12, 16))

        for i in range(4):
            for j in range(3):
                ax = axs[i, j]
                ax.imshow(xs[i + j*4].reshape(28,28), cmap='gray')
                ax.set_title(f'Prediction: {preds[i+j*4].detach().item()} True: {targets[i+j*4]}')
                ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    """
    Run tha whole thang
    """

    train_set, test_set = get_and_process_data()

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, collate_fn=collate_fn, batch_size=16)

    model = ThatMnistNetThatCanTrain(in_dim = 28*28, out_dim=10)
    print("These are my first predictions (I expect them to be shitty)")
    model.plot_predictions(test_loader=test_loader)
    print("TRAINING TEH MODEL")
    model.train(epochs=2, train_data_loader=train_loader, test_data_loader=test_loader)
    total_acc = model.test(test_loader)

    print(f'Yooo, the total accuracy is {total_acc}, thas bretty goooood, frfr skull emoji')
    print("This is what I have after training (they should be slightly better than before)")
    model.plot_predictions(test_loader=test_loader)
