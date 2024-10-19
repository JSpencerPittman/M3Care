import numpy as np
from src.mims.utils import *
import torch
from src.mims.raindrop import Raindrop
from time import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import time
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, 
                             average_precision_score, precision_score, recall_score, f1_score)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
from src.mims.utils import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

warnings.filterwarnings("ignore")

def one_hot(y_):
    # Convert labels to one-hot encoded format
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def generate_global_structure(data, K=10):
    # Generate a global structure using cosine similarity between feature vectors
    observations = data[:, :, :36]
    cos_sim = torch.zeros([observations.shape[0], 36, 36])
    for row in tqdm(range(observations.shape[0])):
        unit = observations[row].T
        cos_sim_unit = cosine_similarity(unit)
        cos_sim[row] = torch.from_numpy(cos_sim_unit)

    # Calculate the average similarity and create a mask for the top-K similar features
    ave_sim = torch.mean(cos_sim, dim=0)
    index = torch.argsort(ave_sim, dim=0)
    index_K = index < K
    global_structure = index_K * ave_sim
    global_structure = masked_softmax(global_structure)
    return global_structure

def diffuse(unit, N=10):
    # Apply diffusion operation to reduce the temporal dimension of the input
    n_time = unit.shape[-1]
    keep = n_time // N - 1
    unit = unit[:, :keep * N].reshape([-1, keep, N])
    return torch.max(unit, dim=-1).values


BATCH_SIZE = 128


def main():
    # Set the manual seed for reproducibility
    torch.manual_seed(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='P19', choices=['P12', 'P19', 'eICU', 'PAM'])
    parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio = 0')
    parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
    parser.add_argument('--reverse', default=False, help='if True, use female, older for training; if False, use female or younger for training')
    parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype == random; otherwise, set as no_removal')
    parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                        help='use this only with P12 dataset (mortality or length of stay)')
    args = parser.parse_args()

    # Set paths for data and model storage
    dataset = args.dataset
    base_path = f'./data/{dataset}'
    model_path = './models/'

    # Define missing ratios to be used during training
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5] if args.withmissingratio else [0]

    for missing_ratio in missing_ratios:
        num_epochs = 20
        learning_rate = 0.0001

        # Define dataset-specific parameters
        if dataset == 'P12':
            d_static, d_inp, max_len, n_classes = 9, 36, 215, 2
        elif dataset == 'P19':
            d_static, d_inp, max_len, n_classes = 6, 34, 60, 2
        elif dataset == 'eICU':
            d_static, d_inp, max_len, n_classes = 399, 14, 300, 2
        elif dataset == 'PAM':
            d_static, d_inp, max_len, n_classes = 0, 17, 600, 8

        # Model hyperparameters
        d_ob = 4
        d_model = d_inp * d_ob
        nhid, nlayers, nhead, dropout = 2 * d_model, 2, 2, 0.2
        aggreg, MAX = 'mean', 100
        n_splits = 5

        # Initialize arrays to store evaluation metrics for each split
        acc_arr = np.zeros(n_splits)
        auprc_arr = np.zeros(n_splits)
        auroc_arr = np.zeros(n_splits)

        for k in range(n_splits):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tb_writer = SummaryWriter(f'./runs/p19/trainer_{timestamp}_{missing_ratio}_{k}')

            split_idx = k + 1
            split_path = f'/splits/phy19_split{split_idx}_new.npy'
            # Load train, validation, and test splits
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=args.splittype, 
                                                                     reverse=args.reverse, baseline=False, 
                                                                     dataset=dataset, predictive_label=args.predictive_label)
            
            # Determine dimensions
            N = len(Ptrain)
            T, F = Ptrain[0]['arr'].shape
            S = len(Ptrain[0]['extended_static'])

            Ptrain_tensor = np.zeros((N, T, F))
            Ptrain_static_tensor = np.zeros((N, S))

            for i in range(N):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            # Calculate mean and stdev for training data
            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

            # Tensorize and normalize all data
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms, ss)

            # Randomly remove features from validation and test sets
            if missing_ratio > 0:
                num_miss_feats = round(missing_ratio * F)
                
                # Remove features randomly on a per sample basis
                if args.feature_removal_level == 'sample':
                    for val_idx in range(len(Pval_tensor)):
                        miss_feat_idxs = np.random.choice(F, num_miss_feats, replace=False)
                        Pval_tensor[val_idx, miss_feat_idxs] = torch.zeros(T, num_miss_feats)
                    for test_idx in range(len(Ptest_tensor)):
                        miss_feat_idxs = np.random.choice(F, num_miss_feats, replace=False)
                        Pval_tensor[test_idx, miss_feat_idxs] = torch.zeros(T, num_miss_feats)

            # Reform input tensors into the shape (T, N, F)             
            Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
            Pval_tensor = Pval_tensor.permute(1, 0, 2)
            Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

            # Reform time tensors into the shape (T, N)
            Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
            Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
            Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

            # Initialize the model
            model = Raindrop(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, d_static, MAX, 0.5, aggreg, n_classes, 
                                global_structure=torch.ones(d_inp, d_inp))
            model = model.cuda()

            # Define loss function, optimizer, and learning rate scheduler
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, 
                                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, 
                                                                   min_lr=1e-8, eps=1e-08, verbose=True)


            # Indices for positive and negative samples
            neg_idxs = (ytrain == 0).squeeze().nonzero()[0]
            pos_idxs = (ytrain == 1).squeeze().nonzero()[0]
            num_neg, num_pos = len(neg_idxs), len(pos_idxs)

            # Expand the number of positive indices
            exp_pos_idxs = np.concatenate([pos_idxs] * 3, axis=0)
            num_pos_exp = len(exp_pos_idxs)

            # Determine number of batches
            k_neg = num_neg // int(BATCH_SIZE / 2)
            k_pos = num_pos_exp // int(BATCH_SIZE / 2)
            num_batches = min(k_neg, k_pos)

            # Metrics to track AUROC, AUPRC, and general loss
            best_val_auroc = 0.0
            print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' %
                   (num_epochs, num_batches, num_epochs * num_batches))

            for epoch in range(num_epochs):
                """Training"""
                model.train()

                np.random.shuffle(exp_pos_idxs)
                np.random.shuffle(neg_idxs)

                for batch in range(num_batches):
                    # Sample both positive and negative batches
                    b_neg_idxs = neg_idxs[batch * int(BATCH_SIZE / 2):(batch + 1) * int(BATCH_SIZE / 2)]
                    b_pos_idxs = exp_pos_idxs[batch * int(BATCH_SIZE / 2):(batch + 1) * int(BATCH_SIZE / 2)]
                    b_idxs = np.concatenate([b_neg_idxs, b_pos_idxs], axis=0)

                    # Extract batch data for training
                    P, Ptime, Pstatic, y = Ptrain_tensor[:, b_idxs, :].cuda(), \
                                           Ptrain_time_tensor[:, b_idxs].cuda(), \
                                           Ptrain_static_tensor[b_idxs].cuda(), \
                                          ytrain_tensor[b_idxs].cuda()

                    lengths = torch.sum(Ptime > 0, dim=0)
                    outputs, _, _ = model(P, Pstatic, Ptime, lengths)

                    # Compute loss, backpropagate, and update model parameters
                    optimizer.zero_grad()
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                # Evaluate training performance
                train_probs = torch.squeeze(torch.sigmoid(outputs))
                train_probs = train_probs.cpu().detach().numpy()
                train_y = y.cpu().detach().numpy()

                train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                train_auprc = average_precision_score(train_y, train_probs[:, 1])

                tb_writer.add_scalar('Loss/train', loss.item(), epoch)
                tb_writer.add_scalar('AUROC/train', train_auroc, epoch)
                tb_writer.add_scalar('AUPRC/train', train_auprc, epoch)

                """Validation"""
                model.eval()
                with torch.no_grad():
                    # Evaluate the model on validation data
                    out_val = evaluate_standard(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor, static=d_static)
                    out_val = torch.squeeze(torch.sigmoid(out_val))
                    out_val = out_val.detach().cpu().numpy()

                    # Calculate validation loss and metrics
                    val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())
                    val_auroc = roc_auc_score(yval, out_val[:, 1])
                    val_auprc = average_precision_score(yval, out_val[:, 1])

                    print("Validation: Epoch %d,  val_loss:%.4f, val_auprc: %.2f, val_auroc: %.2f" %
                        (epoch, val_loss.item(), val_auprc * 100, val_auroc * 100))
                    
                    tb_writer.add_scalar('Loss/val', val_loss.item(), epoch)
                    tb_writer.add_scalar('AUROC/val', val_auroc, epoch)
                    tb_writer.add_scalar('AUPRC/val', val_auprc, epoch)

                    # Adjust learning rate based on validation performance
                    scheduler.step(val_auprc)
                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        torch.save(model.state_dict(), f"{model_path}raindrop_{split_idx}.pt")

            """Testing"""
            model.load_state_dict(torch.load(f"{model_path}raindrop_{split_idx}.pt"))
            model.eval()
    
            with torch.no_grad():
                # Evaluate the model on test data
                out_test = evaluate(model,
                                    Ptest_tensor,
                                    Ptest_time_tensor,
                                    Ptest_static_tensor,
                                    n_classes=n_classes,
                                    static=d_static).numpy()
                ypred = np.argmax(out_test, axis=1)

                # Calculate test metrics
                denoms = np.sum(np.exp(out_test), axis=1).reshape((-1, 1))
                probs = np.exp(out_test) / denoms

                test_acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]
                test_auroc = roc_auc_score(ytest, probs[:, 1])
                test_auprc = average_precision_score(ytest, probs[:, 1])

                print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % 
                      (test_auroc * 100, test_auprc * 100, test_acc * 100))
                print('classification report', classification_report(ytest, ypred))
                print(confusion_matrix(ytest, ypred, labels=list(range(n_classes))))

            # Save results for each split
            acc_arr[k] = test_acc * 100
            auprc_arr[k] = test_auprc * 100
            auroc_arr[k] = test_auroc * 100
        
        # Display mean and standard deviation for evaluation metrics
        mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
        mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
        mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)

        print('------------------------------------------')
        print(f'Accuracy = {mean_acc:.1f} ± {std_acc:.1f}')
        print(f'AUPRC    = {mean_auprc:.1f} ± {std_auprc:.1f}')
        print(f'AUROC    = {mean_auroc:.1f} ± {std_auroc:.1f}')

if __name__ == "__main__":
    main()
