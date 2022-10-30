import torch
import torch.nn as nn

import os
from tqdm import tqdm


def test(name, model, device, test_loader, criterion, epoch, best_acc):
    model.eval()

    correct = 0
    processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    pbar = tqdm(test_loader)

    with torch.no_grad():        
        for batch_id, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device)

            label_pred = model(data)
            label_loss = criterion(label_pred, target)

            # Metrics calculation
            pred = label_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            epoch_loss += label_loss.item()
            pbar.set_description(desc=f'Test Set: Loss={epoch_loss/len(test_loader)}, Batch_id={batch_id}, Test Accuracy={100*correct/processed:0.2f}')

    epoch_accuracy = (100*correct)/processed
    epoch_loss /= len(test_loader)

     # Save checkpoint.
    if epoch_accuracy > best_acc:
        print(f"\n*****Saving Model at epoch: {epoch}*****\n")
        state = {
            'net': model.state_dict(),
            'acc': epoch_accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+ name +'.pth')
        best_acc = epoch_accuracy
    
    return epoch_accuracy, epoch_loss, best_acc
