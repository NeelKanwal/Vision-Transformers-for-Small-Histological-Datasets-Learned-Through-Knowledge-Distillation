""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file contains helpful functions for distallation.py, train_dcnn.py and train_transformer.py mentioned in the paper.
# Update paths to processed datasets

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
font = {'family': 'serif',
        'weight': 'normal',
        'size': 24}
plt.rc('font', **font)
fig = plt.subplots(figsize=(12, 12))

import gpytorch
import pandas as pd
import sys
import numpy as np
import seaborn as sns
import numpy as np
import torch
import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import yagmail


def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
    # print("Distribution of classes: \n", get_class_distribution(natural_img_dataset))
    return count_dict


def convert_batch_list(lst_of_lst):
    return sum(lst_of_lst, [])


# rows to be the “true class” and the columns to be the “predicted class.”
def make_cm(targets_list, predictions_list, classes):
    # labels = [‘True Neg’,’False Pos’,’False Neg’,’True Pos’]
    cm = confusion_matrix(targets_list, predictions_list)
    confusion_matrix_df = pd.DataFrame(cm, columns=classes, index=classes)
    fig = plt.figure(figsize=(12, 10))
    fig = sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="coolwarm")
    fig.set(ylabel="True", xlabel="Predicted", title='DKL predictions')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    return cm


def make_pretty_cm(cf, group_names=None, categories='auto', count=True,
                   percent=True, cbar=True, xyticks=True, xyplotlabels=True, sum_stats=True,
                   figsize=None, cmap='Blues', title=None):
   
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.5)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def train_cnn(model, criterion, optimizer, train_loader, epoch):
    model.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        try:
            output, _, _ = model(data)
        except:
            output, _ = model(data)

        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    # print("Training accuracy: {0:.3f} %\n".format(train_accuracy))
    return train_accuracy, train_loss


def val_cnn(model, early_stopping, timestamp, test_loader, epoch, path, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        stop = False
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            _, preds = torch.max(output, 1)
            # Convert to probabilities if output is logsoftmax
            #  ps = torch.exp(log_ps)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
            # Calculate accuracy
            # equals = pred == targets
            # accuracy = torch.mean(equals)
            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model, epoch, timestamp, path)
        if early_stopping.early_stop:
            # stop_flag_count += 1
            if early_stopping.counter >= early_stopping.patience:
                stop = True
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss, stop

def epoch_test_cnn(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            try:
                output, _ = model(data)
            except:
                output = model(data)

            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            # print(loss)
            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)
            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
        # print(valid_losses)
        valid_loss = np.average(valid_losses)
        return val_accuracy, valid_loss

def extract_features(DenseNetModel, dataloader):
    f = []
    feature = DenseNetModel.features
    # features = torch.nn.Sequential(*list(DenseNetModel.children())[:-1])
    for data, target in dataloader:
        # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        out = feature(data)
        out = F.relu(out, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1) # only works for inputs of 32 x 32
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1)).view(len(data), -1)
        f.append(list(out.detach().cpu().numpy()))
    return f

class custom_classifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.2):
        super(custom_classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # fully connected layer 1
        x = self.dropout(x)
        feat = F.relu(self.fc2(x))  # fully connected layer 2
        x = self.dropout(x)
        x = self.fc3(feat)  # fully connected layer 3
        return x, feat

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target != 0).type(torch.LongTensor).cuda()
            # at = self.alpha.gather(0, target.data.view(-1))
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def infer_cnn(test_loader, model, monte_carlo_runs = 5):
    model.eval()

    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    y_pred, y_true, probs, feature, lower_0c, upper_0c, lower_1c, upper_1c, mean_1 = [], [], [], [], [], [], [], [], []
    for data, target in test_loader:
        temp_p = []
        # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        for i in range(monte_carlo_runs):  # Number of monte carlo simulations for uncertainity

            output, ftr = model(data)
            un, preds = torch.max(output, 1)
            probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
            temp_p.append(probabilities)
        temp_p = np.array(temp_p)
        m_0, s_0 = temp_p[:, :, 0].mean(0), temp_p[:, :, 0].std(0)
        lower_0, upper_0 = m_0 - (s_0 * 1.96) / np.sqrt(5), m_0 + (s_0 * 1.96) / np.sqrt(5)

        m_1, s_1 = temp_p[:, :, 1].mean(0), temp_p[:, :, 1].std(0)
        lower_1, upper_1 = m_1 - (s_1 * 1.96) / np.sqrt(5), m_1 + (s_1 * 1.96) / np.sqrt(5)
        #
        lower_0c.append(list(lower_0))
        upper_0c.append(list(upper_0))
        mean_1.append(list(m_1))
        lower_1c.append(list(lower_1))
        upper_1c.append(list(upper_1))
        probs.append(list(probabilities))
        y_pred.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))
        feature.append(list(ftr.detach().cpu().numpy()))
    return y_pred, y_true, probs, feature, lower_0c, upper_0c, mean_1, lower_1c, upper_1c

def train_simple_transformer(model, criterion, optimizer, train_loader, epoch):
    model.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        try:
            output = model(data)
        except:
            output, _ = model(data)

        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    return train_accuracy, train_loss

def val_simple_transformer(model, early_stopping, timestamp, test_loader, epoch, path, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        stop = False
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            try:
                output = model(data)
            except:
                output, _ = model(data)

            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
            correct += preds.eq(target.view_as(preds)).cpu().sum()
        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model, epoch, timestamp, path)
        if early_stopping.early_stop:
            # stop_flag_count += 1
            if early_stopping.counter >= early_stopping.patience:
                stop = True
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss, stop

def epoch_test_transformer(model, loader, criterion):

    y_pred, y_true, probs = [], [], []
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            _, preds = torch.max(output, 1)
            y_pred.append(list(preds.detach().cpu().numpy()))
            y_true.append(list(target.cpu().numpy()))
            probabilities = F.softmax(output, dim=1)
            probs.append(list(probabilities.detach().cpu().numpy()))

            loss = criterion(output, target)
            # print(loss)

            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)

            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        return y_pred, y_true, probs, val_accuracy, valid_loss

def train_distill_transformer(student, optimizer, train_loader, epoch, distiller):
    student.train()
    distiller.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = student(data)
        _, preds = torch.max(output, 1)

        loss = distiller(data, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    return train_accuracy, train_loss

def val_distill_transformer(student, early_stopping, timestamp, test_loader, epoch, path, distiller):
    with torch.no_grad():
        student.eval()
        distiller.eval()
        valid_losses = []
        correct = 0
        stop = False

        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = student(data)
            _, preds = torch.max(output, 1)
            loss = distiller(data, target)
            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)

            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, student, epoch, timestamp, path)
        if early_stopping.early_stop:
            if early_stopping.counter >= early_stopping.patience:
                stop = True
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss, stop

def epoch_test_dist_transformer(student, val_loader, distiller):
    y_pred, y_true, probs = [], [], []
    with torch.no_grad():
        student.eval()
        valid_losses = []
        correct = 0
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = student(data)
            _, preds = torch.max(output, 1)

            y_pred.append(list(preds.detach().cpu().numpy()))
            y_true.append(list(target.cpu().numpy()))
            probabilities = F.softmax(output, dim=1)
            probs.append(list(probabilities.detach().cpu().numpy()))

            loss = distiller(data, target)
            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)

            correct += preds.eq(target.view_as(preds)).detach().cpu().sum()

        val_accuracy = (100. * correct / float(len(val_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        return y_pred, y_true, probs, val_accuracy, valid_loss


class EarlyStopping_v2:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', timestamp=0000, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.timestamp = timestamp
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch, timestamp, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, timestamp, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, timestamp, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, epoch, timestamp, path):
        path_w = f"{path}/model_checkpoints"
        if not os.path.exists(os.path.join(os.getcwd(), path_w)):
            os.mkdir(os.path.join(os.getcwd(), path_w))
            print("\nDirectory for model checkpoints created.")
        sav_path = f"{path_w}/Epoch:{epoch}_{timestamp}.dat"
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}). \nSaving model to path...{sav_path}')
        state_dict = model.state_dict()
        torch.save({'model': state_dict}, sav_path)
        self.val_loss_min = val_loss


def authenticate_connection():
    # Update following with email and authentication code 
    return yagmail.SMTP('mailsenderaddress@gmail.com', 'xxxxxxxxxxxxxxxxxx')

def sendmail(message):
    import yagmail
    # update these parts here. 
    receiver = "neel.kanwal0@gmail.com"
    # filename = "document.pdf"
    yag = yagmail.SMTP('mailsenderaddress@gmail.com', 'xxxxxxxxxxxxxxxxxx')
    yag.send(
        to=receiver,
        subject="Program Finished...",
        contents=message) # attachments=filename


class SimpleDistiller(nn.Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5):

        super().__init__()

        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha


    def forward(self, img, labels, temperature = None, alpha = None,   **kwargs):
        T = self.temperature
        alpha = self.alpha
        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits = self.student(img)
    
        distill_loss = F.kl_div( F.log_softmax(student_logits / T, dim = -1),
                F.softmax(teacher_logits[0] / T, dim = -1).detach(), reduction = 'batchmean')
        
        distill_loss *= T ** 2
           # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.


        student_loss = F.cross_entropy(student_logits, labels)

        loss = student_loss * alpha + distill_loss * ( 1- alpha)

        return loss
