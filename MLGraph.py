import torch
import numpy as np
from Draw_Eegdata_Physionet_MNE import load_data


def get_graphs(train_data,train_label):
    # membership = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 3, 1, 1,
    #               1, 1, 1, 1, 4, 0, 0, 0, 1, 1, 1, 1, 1, 5, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0,
    #               0, 0, 0, 0, 4, 3, 3, 5, 0, 0, 4, 4, 5, 4, 4, 1]                #0.6210
    # membership = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 3, 1, 1,
    #               4, 1, 1, 1, 5, 0, 0, 0, 1, 1, 1, 1, 1, 6, 0, 1, 0, 4, 2, 0, 6, 4, 6, 6,
    #               6, 5, 5, 5, 5, 3, 3, 6, 5, 5, 5, 5, 6, 5, 5, 1]             #0.6953
    # membership = [0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 4, 1, 1, 1, 1, 3, 0, 5, 2, 2,
    #               6, 2, 2, 2, 7, 1, 1, 1, 2, 2, 2, 2, 2, 4, 7, 4, 7, 6, 3, 1, 7, 6, 7, 7,
    #               7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 2]              #0.6672
    membership = [0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 3, 0, 4, 1, 1, 1, 1, 3, 0, 5, 2, 6,
                  7, 2, 2, 2, 4, 1, 1, 1, 2, 2, 2, 2, 3, 4, 3, 4, 3, 7, 3, 1, 6, 7, 6, 6, 6,
                  6, 5, 5, 5, 5, 5, 6, 6, 5, 5, 5, 6, 6, 5, 2]                 #0.7224
    # membership = [0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 3, 0, 4, 1, 1, 1, 1, 3, 0, 5, 2, 4, 6,
    #               2, 2, 7, 4, 1, 1, 1, 2, 2, 2, 2, 4, 4, 3, 4, 7, 6, 3, 1, 7, 6, 5, 5, 5, 5,
    #               5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2]                      #0.6970

    # vertexs = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    #             'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3',
    #             'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'Ft7',
    #             'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
    #             'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz']

    # Frontal Lobe ：Fp1, Fpz, Fp2, F7, F5, F3, F1, Fz, F2, F4, F6, F8, Af7, Af3, Afz, Af4, Af8, Ft7, Ft8
    # Parietal Lobe：P7, P5, P3, P1, Pz, P2, P4, P6, P8, Cp5, Cp3, Cp1, Cpz, Cp2, Cp4, Cp6
    # Temporal Lobe：T7, T8, T9, T10, Tp7, Tp8
    # Occipital Lobe：O1, Oz, O2, Iz
    # Central Region 电极：C5, C3, C1, Cz, C2, C4, C6
    # Parieto - Occipital电极：Po7, Po3, Poz, Po4, Po8
    # Fronto - Central ：Fc5, Fc3, Fc1, Fcz, Fc2, Fc4, Fc6
    # membership = [6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4,
    #             4, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    #             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #             0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
    #             1, 1, 1, 5, 5, 5, 5, 5, 3, 3, 3, 3]          #10-10

    coff = np.corrcoef(np.squeeze(train_data[0])) - np.identity(64)
    for i in range(1, len(train_data)):
        coff += (np.corrcoef(np.squeeze(train_data[i])) - np.identity(64))
    coff = coff / len(train_data)

    subGraphs = np.zeros((8, 64, 64), dtype=np.float32)
    for k in range(0, 64):
        m = k + 1
        if membership[k] == 0:
            for n in range(m, 64):
                if membership[n] == 0:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[0][k][n] = coff[k][n]
                        subGraphs[0][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 1:
            for n in range(m, 64):
                if membership[n] == 1:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[1][k][n] = coff[k][n]
                        subGraphs[1][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 2:
            for n in range(m, 64):
                if membership[n] == 2:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[2][k][n] = coff[k][n]
                        subGraphs[2][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 3:
            for n in range(m, 64):
                if membership[n] == 3:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[3][k][n] = coff[k][n]
                        subGraphs[3][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 4:
            for n in range(m, 64):
                if membership[n] == 4:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[4][k][n] = coff[k][n]
                        subGraphs[4][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 5:
            for n in range(m, 64):
                if membership[n] == 5:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[5][k][n] = coff[k][n]
                        subGraphs[5][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 6:
            for n in range(m, 64):
                if membership[n] == 6:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[6][k][n] = coff[k][n]
                        subGraphs[6][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
        if membership[k] == 7:
            for n in range(m, 64):
                if membership[n] == 7:
                    if coff[k][n] == 0:
                        continue
                    else:
                        subGraphs[7][k][n] = coff[k][n]
                        subGraphs[7][n][k] = coff[k][n]
                        coff[k][n] = 0
                        coff[n][k] = 0
    local_graph_weight = subGraphs[0] + subGraphs[1] + subGraphs[2] + subGraphs[3] + subGraphs[4] + subGraphs[5] + subGraphs[6] + subGraphs[7]
    local_graph_weight = local_graph_weight + np.identity(64)
    local_graph_weight = local_graph_weight.reshape(-1).astype('float32')
    local_graph_weight = torch.squeeze(torch.tensor(local_graph_weight,device='cuda'))

    edge = torch.empty((2, 0), dtype=torch.long, device='cuda')
    for raw in range(64):
        for col in range(64):
            edge = torch.hstack((edge, torch.tensor([[raw], [col]], dtype=torch.long, device='cuda')))

    global_mask = coff + np.identity(64)
    global_mask = (global_mask > 0).astype(int)
    global_mask = torch.squeeze(torch.tensor(global_mask, device='cuda'))

    return edge,local_graph_weight,global_mask