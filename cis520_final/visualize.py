#Script to load model, predict sequences and plot
# import relevant libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import loader
import os

cur_dataset = 'eth' #args.dataset_name

data_dir = os.path.join('/home/roongtaaahsih/ped_traj/sgan_ab/scripts/datasets/', cur_dataset + '/test')
# load trained model
gru_net = torch.load('./saved_models/gru_model_eth_lr_0.0017_epoch_100_predlen_8.pt')
gru_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference
# test function to calculate and return avg test loss after each epoch
def test(gru_net,pred_len,data_dir):

    test_data_dir = data_dir #os.path.join('/home/ashishpc/Desktop/sgan_ab/scripts/datasets/', cur_dataset + '/train')

    # retrieve dataloader
    _, dataloader = loader.data_loader(test_data_dir)

    # define parameters for training and testing loops
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []
    test_avgD_error=[]
    test_finalD_error=[]
    plt.figure(figsize=(32,20))
    plt.xlabel("X coordinates of pedestrians")
    plt.ylabel("Y coordinates of pedestrians")
    # now, test the model
    for i, batch in enumerate(dataloader):
        test_observed_batch = batch[0]
        test_target_batch = batch[1]
        out = gru_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        # cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        # test_loss.append(cur_test_loss.item())
        cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        test_loss.append(cur_test_loss.item())

        s,peds,c=out.shape
        out1=out.detach().numpy()
        target1=test_target_batch.detach().numpy()
        observed1=test_observed_batch.detach().numpy()
        print("observed 1 shape:",observed1.shape)
        print("target1 shape:", target1.shape)
        print("out 1 shape", out1.shape)
        out2=np.vstack((observed1,out1))
        _=np.vstack((observed1,target1))
        print("out2 shape",out2.shape)
        avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0]-target1[:,:,0])+
            np.square(out1[:,:,1]-target1[:,:,1]))))/(pred_len*peds)
        test_avgD_error.append(avgD_error)

        # final displacement error
        finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0]-target1[pred_len-1,:,0])+
            np.square(out1[pred_len-1,:,1]-target1[pred_len-1,:,1]))))/peds
        test_finalD_error.append(finalD_error)
                
    avg_testloss = sum(test_loss)/len(test_loss)
    avg_testD_error=sum(test_avgD_error)/len(test_avgD_error)
    avg_testfinalD_error=sum(test_finalD_error)/len(test_finalD_error)

        # out1=out
        # target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
        # seq, peds, coords = test_target_batch.shape

    return avg_testloss,avg_testD_error,avg_testfinalD_error

def main():
    
    '''define parameters for training and testing loops!'''

    pred_len = 8
    
    # retrieve dataloader
    _, _ = loader.data_loader(data_dir)

    ''' define the network, optimizer and criterion '''

    #calling the test function and calculating the test losses
    avg_testloss,avg_testD_error,avg_testfinalD_error=test(gru_net,pred_len,data_dir)
    textfilename="./txtfiles/"+"gru_results_100epoch_lr0.0017"
    with open(textfilename,"a+") as f:
        f.write(str(pred_len)+"\t"+str(avg_testloss)+"\t"+str(avg_testD_error)+"\t"+str(avg_testfinalD_error)+"\n")
    pass

'''main function'''
if __name__ == '__main__':
    main()