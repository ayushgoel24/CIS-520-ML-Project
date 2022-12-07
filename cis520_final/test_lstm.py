# import relevant libraries
import torch
import torch.nn as nn
import loader

# load trained model
vanilla_lstm_net = torch.load('./saved_models/vanilla_lstm_model_lr_0.0001_epoch_1_predlen_8.pt')
vanilla_lstm_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference

def test():
    test_data_dir = "./datasets/eth/test"

    # retrieve dataloader
    _, dataloader = loader.data_loader(test_data_dir)

    # define parameters for training and testing loops
    pred_len = 12
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []

    # now, test the model
    for _, batch in enumerate(dataloader):
      test_observed_batch = batch[0]
      test_target_batch = batch[1]
      out = vanilla_lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
      print("out's shape:", out.shape)
      cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
      print('Current test loss: {}'.format(cur_test_loss.item())) # print current test loss
      test_loss.append(cur_test_loss.item())
    avg_testloss = sum(test_loss)/len(test_loss)
    print("========== Average test loss:", avg_testloss, "==========")

    pass

def main():
    test() # test all of the data in test_data_dir
    print("Done testing!")

if __name__ == '__main__':
    main()