import math
import numpy
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from Draw_Eegdata_Physionet_MNE import load_data
from Layers import MHDGcnNet
from MLGraph import get_graphs
from sklearn.metrics import roc_auc_score,f1_score

if __name__ == "__main__":
    print("load data！")
    total_data,total_labels = load_data()
    print(total_data)
    print(total_labels)
    print("data collected！")

    edges,local_graph_weight,global_mask = get_graphs(total_data,total_labels)

    total_data_len = len(total_data)

    args = {
        "lr": 0.005,
        "dropout": 0.55,
        "batch_size": 168,
        "decay": 0.001
    }

    avg_accuracy = 0
    avg_std = []
    AUC_std = []
    avg_AUC = 0

    for i in range(0,10):
        test_data = total_data[i*10*42:(i+1)*10*42]
        train_data = np.vstack((total_data[:i*10*42],total_data[(i+1)*10*42:]))
        test_labels = total_labels[i*10*42:(i+1)*10*42]
        train_labels = np.hstack((total_labels[:i*10*42],total_labels[(i+1)*10*42:]))

        train_data_len = 94*42
        test_data_len = len(test_data)

        np.random.shuffle(train_data)
        np.random.shuffle(train_labels)
        np.random.shuffle(test_data)
        np.random.shuffle(test_labels)

        train_data = torch.tensor(train_data).type(torch.FloatTensor)
        train_labels = torch.tensor(train_labels).type(torch.LongTensor)
        test_data = torch.tensor(test_data).type(torch.FloatTensor)
        test_labels = torch.tensor(test_labels).type(torch.LongTensor)

        x_train = DataLoader(train_data,batch_size=args["batch_size"],shuffle=False)
        y_train = DataLoader(train_labels,batch_size=args["batch_size"],shuffle=False)
        x_test = DataLoader(test_data,batch_size=args["batch_size"],shuffle=False)
        y_test = DataLoader(test_labels,batch_size=args["batch_size"],shuffle=False)

        model = MHDGcnNet(160,args["dropout"]).cuda()

        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"],weight_decay=args["decay"])

        for epochs in range(1,201):
            total_accuracy = 0
            print("------epoch {} training begins------  ".format(epochs))
            train_total_loss = 0

            model.train()
            for data,labels in zip(x_train,y_train):
                optimizer.zero_grad()
                output= model(data.cuda(),edges,local_graph_weight,global_mask)
                loss = loss_fn(output, labels.cuda())
                train_total_loss = train_total_loss + loss

                loss.backward()
                optimizer.step()

                step_accuracy = (output.argmax(1) == labels.cuda()).sum()
                total_accuracy = total_accuracy + step_accuracy
                print("step accuracy",step_accuracy/args["batch_size"])
            print("total accuracy",total_accuracy/train_data_len)
            print("epoch {} , total loss:{}".format(epochs, train_total_loss))

            total_test_loss = 0 
            test_accuracy = 0
            prob_all = []
            label_all = []
            y_pred = []

            model.eval()
            with torch.no_grad():
                for data,labels in zip(x_test,y_test):

                    output = model(data.cuda(),edges,local_graph_weight,global_mask)
                    loss = loss_fn(output, labels.cuda())

                    predict_labels = output.argmax(1)
                    step_accuracy = (predict_labels == labels.cuda()).sum()
                    test_accuracy = step_accuracy + test_accuracy

                    total_test_loss = total_test_loss + loss.item()

                    prob_all.extend(output[:, 1].cpu().numpy())
                    label_all.extend(labels.cpu().numpy())
                    y_pred.extend(output.argmax(1).cpu().numpy())

            roc_auc_scores = roc_auc_score(label_all, prob_all)
            print("epoch {} , test_accuracy: {}  ,  test_loss:{}    AUC:{:.4f}    F1 score:{:.4f}".format(epochs, test_accuracy / test_data_len,total_test_loss,
                            roc_auc_scores,f1_score(label_all,y_pred,average='macro')))
