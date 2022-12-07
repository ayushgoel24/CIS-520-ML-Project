#including relu activaton and dropout after the first linear layer
# prototype of lstm network for pedestrian modeling
# written by: Ashish Roongta, Fall 2018
# carnegie mellon university

# import relevant libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import loader
import os
import matplotlib.pyplot as plt 

cur_dataset = 'eth'
use_cuda = False
num_epoch_ = 20
pred_len_ = 12
learning_rate_ = 0.003
obs_len_ = 8

data_dir = os.path.join('./train')

''' Class for defining the lstm Network '''
class VanillaLSTMNet(nn.Module):
    def __init__(self):
        super(VanillaLSTMNet, self).__init__()
        
        ''' Inputs to the lstmCell's are (input, (h_0, c_0)):
         1. input of shape (batch, input_size): tensor containing input 
         features
         2a. h_0 of shape (batch, hidden_size): tensor containing the 
         initial hidden state for each element in the batch.
         2b. c_0 of shape (batch, hidden_size): tensor containing the 
         initial cell state for each element in the batch.
        
         Outputs: h_1, c_1
         1. h_1 of shape (batch, hidden_size): tensor containing the next 
         hidden state for each element in the batch
         2. c_1 of shape (batch, hidden_size): tensor containing the next 
         cell state for each element in the batch '''
        
        # set parameters for network architecture
        self.embedding_size = 64
        self.rnn_size=128
        self.input_size = 2
        self.output_size = 2
        self.dropout_prob = 0.5
        if(use_cuda):
            self.device = torch.device("cuda:0") # to run on GPU
        else:
            self.device=torch.device("cpu")

        # linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # define lstm cell
        self.lstm_cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        # linear layer to map the hidden state of lstm to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        
        pass
 
    def forward(self, observed_batch, pred_len = 0):
        ''' this function takes the input sequence and predicts the output sequence. 
        
            args:
                observed_batch (torch.Tensor) : input batch with shape <seq length x num pedestrians x number of dimensions>
                pred_len (int) : length of the sequence to be predicted.

        '''
        output_seq = []

        ht = torch.zeros(observed_batch.size(1), self.rnn_size,device=self.device, dtype=torch.float)
        ct = torch.zeros(observed_batch.size(1), self.rnn_size,device=self.device, dtype=torch.float)
        seq, peds, coords = observed_batch.shape

        # feeding the observed trajectory to the network
        for step in range(seq):
            observed_step = observed_batch[step, :, :]
            lin_out = self.input_embedding_layer(observed_step.view(peds,2))
            input_embedded=self.dropout(self.relu(lin_out))
            ht,ct = self.lstm_cell(input_embedded, (ht,ct))
            out = self.output_layer(ht)

        # getting the predicted trajectory from the pedestrian 
        for i in range(pred_len):
            lin_out = self.input_embedding_layer(out)
            input_embedded=self.dropout(self.relu(lin_out))
            ht,ct = self.lstm_cell(input_embedded, (ht,ct))
            out = self.output_layer(ht)
            output_seq += [out]
            
        output_seq = torch.stack(output_seq).squeeze() # convert list to tensor
        return output_seq

# test function to calculate and return avg test loss after each epoch
def test(vanilla_lstm_net,pred_len=0):

    test_data_dir = os.path.join('./test')

    # retrieve dataloader
    dataset, dataloader = loader.data_loader(test_data_dir)

    # define parameters for training and testing loops
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []
    test_avgD_error=[]
    test_finalD_error=[]
    num_test_peds=0 #counter for number of pedestrians in the test dataset
    # now, test the model
    for i, batch in enumerate(dataloader):
        if(use_cuda): #when use_cuda has been mentioned in the argument
            test_observed_batch = batch[0].cuda()
            test_target_batch = batch[1].cuda()
        else: #when use_cuda has not been mentioned in the argument
            test_observed_batch = batch[0]
            test_target_batch = batch[1]

        out = vanilla_lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        test_loss.append(cur_test_loss.item())
        out1=out
        target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
        if(use_cuda): #to transfer the tensors back to CPU if cuda was used
            out1=out1.cpu()
            target_batch1=target_batch1.cpu()

        seq, peds, coords = test_target_batch.shape
        num_test_peds+=peds
        avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
            np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
        test_avgD_error.append(avgD_error)

        # final displacement error
        finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
            np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
        test_finalD_error.append(finalD_error)
                
    avg_testloss = sum(test_loss)/len(test_loss)
    avg_testD_error=sum(test_avgD_error)/len(test_avgD_error)
    avg_testfinalD_error=sum(test_finalD_error)/len(test_finalD_error)
    print("============= Average test loss:", avg_testloss, "====================")


    return avg_testloss, avg_testD_error,avg_testfinalD_error,num_test_peds



def main():
    
    '''define parameters for training and testing loops!'''

    num_epoch = num_epoch_
    pred_len = pred_len_
    learning_rate = learning_rate_
    obs_len = obs_len_
    # retrieve dataloader
    _, dataloader = loader.data_loader(data_dir)

    ''' define the network, optimizer and criterion '''
    name=cur_dataset # to add to the name of files
    vanilla_lstm_net = VanillaLSTMNet()
    if(use_cuda):
        vanilla_lstm_net.cuda()

    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
    optimizer = optim.Adam(vanilla_lstm_net.parameters(), lr=learning_rate)

    # initialize lists for capturing losses/errors
    train_loss = []
    test_loss = []
    avg_train_loss = []
    avg_test_loss = []
    train_avgD_error=[]
    train_finalD_error=[]
    avg_train_avgD_error=[]
    avg_train_finalD_error=[]
    test_finalD_error=[]
    test_avgD_error=[]
    std_train_loss = []
    num_train_peds=[] #counter for number of pedestrians taken for each epoch


    '''training loop'''
    for i in range(num_epoch):
       
        print('======================= Epoch: {cur_epoch} / {total_epochs} =======================\n'.format(cur_epoch=i, total_epochs=num_epoch))
        def closure():
            train_peds=0 #counter for number of pedestrians taken for each epoch
            for i, batch in enumerate(dataloader):
                if(use_cuda):
                    train_batch = batch[0].cuda()
                    target_batch = batch[1].cuda()
                else:
                    train_batch = batch[0]
                    target_batch = batch[1]

                # print("train_batch's shape", train_batch.shape)
                # print("target_batch's shape", target_batch.shape)
                seq, peds, coords = train_batch.shape # q is number of pedestrians 
                train_peds+=peds #keeping a count of the number of peds in the data
                out = vanilla_lstm_net(train_batch, pred_len=pred_len) # forward pass of lstm network for training
                # print("out's shape:", out.shape)
                optimizer.zero_grad() # zero out gradients
                cur_train_loss = criterion(out, target_batch) # calculate MSE loss
                # print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss
                print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss
                
                #calculating average deisplacement error
                out1=out
                target_batch1=target_batch  #making a copy of the tensors to convert them to array
                if(use_cuda):
                    out1=out1.cpu()
                    target_batch1=target_batch1.cpu()
                
                avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
                    np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
                train_avgD_error.append(avgD_error)

                #calculate final displacement error
                finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
                    np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
                train_finalD_error.append(finalD_error)

                train_loss.append(cur_train_loss.item())
                cur_train_loss.backward() # backward prop
                optimizer.step() # step like a mini-batch (after all pedestrians)
            num_train_peds.append(train_peds)
            return cur_train_loss
        optimizer.step(closure) # update weights

        # save model at every epoch (uncomment) 
        # torch.save(lstm_net, './saved_models/lstm_model_v3.pt')
        # print("Saved lstm_net!")
        avg_train_loss.append(np.sum(train_loss)/len(train_loss))
        avg_train_avgD_error.append(np.sum(train_avgD_error)/len(train_avgD_error))
        avg_train_finalD_error.append(np.sum(train_finalD_error)/len(train_finalD_error))   
        std_train_loss.append(np.std(np.asarray(train_loss)))
        train_loss = [] # empty train loss

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("average train loss: {}".format(avg_train_loss))
        print("average std loss: {}".format(std_train_loss))
        avgTestLoss,avgD_test,finalD_test,num_test_peds=test(vanilla_lstm_net,pred_len)
        avg_test_loss.append(avgTestLoss)
        test_finalD_error.append(finalD_test)
        test_avgD_error.append(avgD_test)
        print("test finalD error: ",finalD_test)
        print("test avgD error: ",avgD_test)


    '''after running through epochs, save your model and visualize.
       then, write your average losses and standard deviations of 
       losses to a text file for record keeping.'''

    save_path = os.path.join('./saved_models/', 'lstm_model_'+name+'_lr_' + str(learning_rate) + '_epoch_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+ '.pt')
    # torch.save(lstm_net, './saved_models/lstm_model_lr001_ep20.pt')
    torch.save(vanilla_lstm_net, save_path)
    print("saved lstm_net! location: " + save_path)

    ''' visualize losses vs. epoch'''
    plt.figure() # new figure
    plt.title("Average train loss vs {} epochs".format(num_epoch))
    plt.plot(avg_train_loss,label='avg train_loss') 
    plt.plot(avg_test_loss,color='red',label='avg test_loss')
    plt.legend()
    plt.savefig("./saved_figs/" + "lstm_"+name+"_avgtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+  '.png')
    # plt.show()
    # plt.show(block=True)
    
    plt.figure() # new figure
    plt.title("Average and final displacement error {} epochs".format(num_epoch))
    plt.plot(avg_train_finalD_error,label='train:final disp. error') 
    plt.plot(avg_train_avgD_error,color='red',label='train:avg disp. error')
    plt.plot(test_finalD_error,color='green',label='test:final disp. error')
    plt.plot(test_avgD_error,color='black',label='test:avg disp. error')
    plt.ylim((0,10))
    plt.legend()
    # plt.show()
    plt.savefig("./saved_figs/" + "lstm_"+name+"_avg_final_displacement_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+  '.png')

    plt.figure()
    plt.title("Std of train loss vs epoch{} epochs".format(num_epoch))
    plt.plot(std_train_loss)
    plt.savefig("./saved_figs/" + "lstm_"+name+"_stdtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+'.png')
    # plt.show(block=True)
    print("saved images for avg training losses! location: " + "./saved_figs")

    # save results to text file
    txtfilename = os.path.join("./txtfiles/", "lstm_"+name+"_avgtrainlosses_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename, "w") as f:
        f.write("Number of pedestrians in the traning data: {}\n".format(num_train_peds[-1]))
        f.write("Number of pedestrians in the test dataset: {}\n".format(num_test_peds))
        f.write("\n==============Average train loss vs. epoch:===============\n")
        f.write(str(avg_train_loss))
        f.write("\nepochs: " + str(num_epoch))
        f.write("\n==============Std train loss vs. epoch:===================\n")
        f.write(str(std_train_loss))
        f.write("\n==============Avg test loss vs. epoch:===================\n")
        f.write(str(avg_test_loss))
        f.write("\n==============Avg train displacement error:===================\n")
        f.write(str(avg_train_avgD_error))
        f.write("\n==============Final train displacement error:===================\n")
        f.write(str(avg_train_finalD_error))
        f.write("\n==============Avg test displacement error:===================\n")
        f.write(str(test_avgD_error))
        f.write("\n==============Final test displacement error:===================\n")
        f.write(str(test_finalD_error))
    print("saved average and std of training losses to text file in: ./txtfiles")

    # saving all data files with different prediction lengths and observed lengths for each dataset    
    txtfilename2 = os.path.join("./txtfiles/", "RESULTS_LSTM2"+"_diff_ObsPred_len_lr_"+ str(learning_rate) + '_numepochs_' + str(num_epoch)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename2,"a+") as g: #opening the file in the append mode
        if(pred_len==2):
            g.write("Dataset: "+name+" ;Number of epochs: {}".format(num_epoch)+"\n")
            g.write("obs_len"+"\t"+"pred_len"+"\t"+"avg_train_loss"+"\t"+"avg_test_loss"+"\t"+"std_train_loss"+"\t"
            	+"avg_train_dispacement"+"\t"+"final_train_displacement"+"\t"+"avg_test_displacement"+"\t"
            	+"final_test_displacement"+"\t"+"Num_train_peds"+"\t"+"Num_Test_peds"+"\n")
        # outputing the current observed length
        g.write(str(obs_len)+"\t")
        # outputing the current prediction length
        g.write(str(pred_len)+"\t")
        #the avg_train_loss after total epochs
        g.write(str(avg_train_loss[-1])+"\t")
        # the avg_test_loss after total epochs
        g.write(str(avg_test_loss[-1])+"\t")
        # the standard deviation of train loss
        g.write(str(std_train_loss[-1])+"\t")
        # the avg train dispacement error
        g.write(str(avg_train_avgD_error[-1])+"\t")
        # the train final displacement error
        g.write(str(avg_train_finalD_error[-1])+"\t")
        # the test avg displacement error
        g.write(str(test_avgD_error[-1])+"\t")
        # the test final displacement error
        g.write(str(test_finalD_error[-1])+"\t")
        # the number of pedestrians in the traininig dataset
        g.write(str(num_train_peds[-1])+"\t")
        # Number of pedestrian sin the training dataset
        g.write(str(num_test_peds)+"\n")
    print("saved all the results to the text file for observed length: {}".format(obs_len))

'''main function'''
if __name__ == '__main__':
    main()

