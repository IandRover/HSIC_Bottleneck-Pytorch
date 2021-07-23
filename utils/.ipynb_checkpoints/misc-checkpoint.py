import torch
from torch import nn
from torchvision import datasets, transforms

def get_filename(args):
    if args.BP == 1:
        filename = "{}_{}_{}".format(args.dataset, args.model, args.loss)
    else:
        filename = "{}_{}_{}_F{}_{}{}{}_S{}L{}".format(args.dataset, args.model, args.loss, args.forward, args.kernel_x[0], args.kernel_h[0], args.kernel_y[0], args.sigma_, args.lambda_, )
    if args.bn_affine == 1:
        filename += "_bn"
    if args.Latinb == 1:
        try:
            filename += "_Latinb{}{}".format(args.Latinb_type, args.Latinb_lambda)
        except:
            filename += "_Latinb{}".format(args.Latinb_lambda)
    return filename

def show_result(hsic, train_loader, test_loader, epoch, logs, device):
    hsic.model.eval()
    with torch.no_grad():
        counts, correct, counts2, correct2 = 0, 0, 0, 0        
        for batch_idx, (data, target) in enumerate(train_loader): 
            output = hsic.model.forward(data.view(data.size(0), -1).to(device))[0].cpu()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred[:,0] == target).float().sum()
            counts += len(pred)
        for batch_idx, (data, target) in enumerate(test_loader): 
            output = hsic.model.forward(data.view(data.size(0), -1).to(device))[0].cpu()
            pred = output.argmax(dim=1, keepdim=True)
            correct2 += (pred[:,0] == target).float().sum()
            counts2 += len(pred)
            
        train_acc = correct/counts
        test_acc = correct2/counts2
        print("EPOCH {}. \t Training  ACC: {:.4f}. \t Testing ACC: {:.4f}".format(epoch, train_acc, test_acc))
        logs.append([epoch, train_acc.numpy(), test_acc.numpy()])