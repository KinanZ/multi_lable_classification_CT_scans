from tqdm import tqdm
import torch
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss


# validation function
def validate(model, dataloader, criterion, val_data, device, evaluate=False):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    model_results = []
    targets = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['labels'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
            if evaluate:
                model_results.extend(outputs.cpu().numpy())
                targets.extend(target.cpu().numpy())

        val_loss = val_running_loss / counter
        if evaluate:
            result = calculate_metrics(np.array(model_results), np.array(targets))
            return val_loss, result
        else:
            return val_loss


def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    # Accuracy: In multilabel classification, this function computes subset accuracy: the set of labels predicted
    # for a sample must exactly match the corresponding set of labels in y_true.
    accuracy = accuracy_score(y_true=target, y_pred=pred)

    # 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro': Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account.
    # 'samples': Calculate metrics for each instance, and find their average
    # (only meaningful for multilabel classification where this differs from accuracy_score)

    micro_precision = precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0)

    macro_precision = precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0)
    macro_f1 =f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0)

    samples_precision = precision_score(y_true=target, y_pred=pred, average='samples', zero_division=0)
    samples_recall = recall_score(y_true=target, y_pred=pred, average='samples', zero_division=0)
    samples_f1 = f1_score(y_true=target, y_pred=pred, average='samples', zero_division=0)

    return {
        'accuracy': accuracy,
        'micro/precision': micro_precision,
        'micro/recall': micro_recall,
        'micro/f1': micro_f1,
        'macro/precision': macro_precision,
        'macro/recall': macro_recall,
        'macro/f1': macro_f1,
        'samples/precision': samples_precision,
        'samples/recall': samples_recall,
        'samples/f1': samples_f1,
            }
