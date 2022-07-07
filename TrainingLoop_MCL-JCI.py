import time
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SiameseNet import Siamese
import DataLoader_MCLJCI as DataLoad
from coeff_func import *


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def createLossAndOptimizer(net, learning_rate):
    # Loss function as defined above
    loss = ContrastiveLoss().cuda()
    # Adam Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return loss, optimizer


model = Siamese().cuda()


def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)

    # Get pair information with ground truth dissimilarity metrics from the provided JND file.
    jnd_pairs = np.genfromtxt(open('jnd_pairs_npy.csv'), delimiter=',', dtype=str)

    # Split dataset into 3 parts as training, validation and test.(total number of pairs=4950)
    train_split = 4000
    val_split = 4500

    # Set root directory for training data
    training_datadir = './P_map_npyfiles_hdrvdp3_detection'

    # call Dataloader function with defined splits.
    data_train = DataLoad.mcl_jci(jnd_pairs[:train_split], root_dir=training_datadir)
    data_val = DataLoad.mcl_jci(jnd_pairs[train_split:val_split], root_dir=training_datadir)
    data_test = DataLoad.mcl_jci(jnd_pairs[val_split:], root_dir=training_datadir)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()
    # calculate number of batches according to split size and batch size
    n_batches = train_split // batch_size

    # initiate validation loss with a high enough value.
    last_valloss = 1000

    # Loop for number of epochs
    for epoch in range(n_epochs):

        # initiate parameters for statistic recordings.
        dist = []
        y_true = []
        running_loss = 0.0
        total_train_loss = 0
        start_time = time.time()
        print_every = 5

        for i, data in enumerate(data_train_loader, 0):
            # Get input images and ground truth dissimilarity score from the training dataloader
            refs, tests, gts = data
            # store ground truth values in cpu for correlation calculations
            y_val = gts.numpy()
            # Wrap the tensors in a Variable object
            refs, tests, gts = Variable(refs).type(torch.cuda.FloatTensor), Variable(tests).type(torch.cuda.FloatTensor), Variable(gts).type(torch.cuda.FloatTensor)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            output1, output2 = net(refs, tests)
            loss_size = loss(output1, output2, gts)
            loss_size.backward()
            optimizer.step()

            # Store acquired loss on the initialized parameters
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            # Store prediction values for future correlation calculation.
            euclidean_distance = F.pairwise_distance(output1, output2).cpu().detach().numpy()
            for elm in euclidean_distance:
                dist.append(elm)
            for elm in y_val:
                y_true.append(elm)

            # Print every "print_every"-th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.6f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # Calculate correlation coefficients between the predicted values and ground truth values on training set.
        dist = np.array(dist)
        y_true = np.array(y_true)
        _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist, y_true)
        print("Training set: PCC{:.4}, SROCC{:.4}, KROCC{:.4}, RMSE{:.4}".format(cc_v, srocc_v, krocc_v, rmse_v))

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        val_counter = 0
        dist = []
        y_true = []
        for _, data in enumerate(data_val_loader, 0):
            with torch.no_grad():
                # Get input images and ground truth dissimilarity score from the validation dataloader
                val_refs, val_tests, val_gts = data
                # store ground truth values in cpu for correlation calculations
                y_val = val_gts.numpy()
                # Wrap them in a Variable object
                val_refs, val_tests, val_gts = Variable(val_refs).type(torch.cuda.FloatTensor), Variable(val_tests).type(
                    torch.cuda.FloatTensor), Variable(val_gts).type(torch.cuda.FloatTensor)

                # Forward pass
                val_out1, val_out2 = net(val_refs, val_tests)
                euclidean_distance = F.pairwise_distance(val_out1, val_out2).cpu().detach().numpy()
                loss_size = loss(val_out1, val_out2, val_gts)
                total_val_loss += loss_size.cpu().numpy()
                val_counter += 1

                # Store prediction values for future correlation calculation.
                for elm in euclidean_distance:
                    dist.append(elm)
                for elm in y_val:
                    y_true.append(elm)

                # Save model parameters if it validation loss is lower than previous best validation loss
                if (total_val_loss / val_counter) < last_valloss:
                    last_valloss = total_val_loss / val_counter
                    torch.save(model.state_dict(), 'ModelParams_BestVal.pt')

        # Calculate correlation coefficients between the predicted values and ground truth values on validation set.
        dist = np.array(dist)
        y_true = np.array(y_true)
        _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist, y_true)
        print("Validation set: PCC{:.4}, SROCC{:.4}, KROCC{:.4}, RMSE{:.4}".format(cc_v, srocc_v, krocc_v, rmse_v))
        print("Validation loss = {:.6}".format(total_val_loss / val_counter))

    # After reaching max epochs, do a single loop over test dataset.
    total_test_loss = 0
    test_counter = 0
    for i, data in enumerate(data_test_loader, 0):
        with torch.no_grad():
            test_counter += 1
            test_refs, test_tests, test_gts = data
            test_refs, test_tests, test_gts = Variable(test_refs).type(torch.cuda.FloatTensor), Variable(test_tests).type(
                torch.cuda.FloatTensor), Variable(test_gts).type(torch.cuda.FloatTensor)
            # test_refs, test_tests = test_refs.permute(0, 3, 1, 2), test_tests.permute(0, 3, 1, 2)

            test_out1, test_out2 = net(test_refs, test_tests)
            test_loss_size = loss(test_out1, test_out2, test_gts)
            total_test_loss += test_loss_size.item()

    # Print final statistics and save model parameters.
    print("Test loss = {:.6f}".format(total_test_loss / test_counter))
    torch.save(model.state_dict(), 'ModelParams_MaxEpoch.pt')
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


# Call training function with corresponding batch size, number of epochs and learning rate.
trainNet(model, batch_size=32, n_epochs=100, learning_rate=0.02)
