#Script to load model, predict sequences and plot
# import relevant libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
import loader
import os

cur_dataset = 'eth' #args.dataset_name

data_dir = os.path.join('./test')
# load trained model
lstm_net = torch.load('./saved_models/vanilla_lstm_model_lr_0.0017_epoch_100_predlen_8_batchsize_5.pt')
lstm_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference

# test function to calculate and return avg test loss after each epoch
def test(lstm_net,pred_len,data_dir):

    test_data_dir = data_dir #os.path.join('/home/ashishpc/Desktop/sgan_ab/scripts/datasets/', cur_dataset + '/train')

    # retrieve dataloader
    _, dataloader = loader.data_loader(test_data_dir)

    plt.figure(figsize=(32,20))
    plt.xlabel("X coordinates of pedestrians")
    plt.ylabel("Y coordinates of pedestrians")
    # now, test the model
    for _, batch in enumerate(dataloader):
        test_observed_batch = batch[0]
        test_target_batch = batch[1]
        out = lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        s,_,_=out.shape
        out1=out.detach().numpy()
        target1=test_target_batch.detach().numpy()
        observed1=test_observed_batch.detach().numpy()
        print("observed 1 shape:",observed1.shape)
        print("target1 shape:", target1.shape)
        print("out 1 shape", out1.shape)
        out2=np.vstack((observed1,out1))
        target2=np.vstack((observed1,target1))
        print("out2 shape",out2.shape)
        for t in range(6):
            plt.plot(observed1[:,t,0],observed1[:,t,1],color='b',marker='o',linewidth=5,markersize=12)
            plt.plot(target2[s-1:s+pred_len,t,0],target2[s-1:s+pred_len,t,1],color='red',marker='o',linewidth=5,markersize=12)
            plt.plot(out2[s-1:s+pred_len,t,0],out2[s-1:s+pred_len,t,1],color='g',marker='o',linewidth=5,markersize=12)
        plt.legend(["Observed","Ground Truth","Predicted"])
        plt.show(block=True)

def main(args):
    
    '''define parameters for training and testing loops!'''
    pred_len = 8 #args.pred_len
    test(lstm_net,pred_len,data_dir)
    pass

'''main function'''
if __name__ == '__main__':
    main()