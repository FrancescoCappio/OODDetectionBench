import torch
import numpy as np
from tqdm import tqdm

from models.evaluators.common import prepare_ood_labels, calc_ood_metrics


def mahalanobis_evaluator(train_loader, test_loader, device, model):
    # implements neurips 2018: https://papers.nips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html

    # the mahalanobis distance is computed not for a single network layer, but for a 
    # set of network layers (see model_feature_list function)

    # remember that this is not a per class mahalanobis, in the sense that we compute a single covariance matrix. 
    # In practice we are assuming that the features covariance is the same for all classes, while of course the 
    # features mean is different for each class. 
    # This means that in practice we are not computing statistics over the whole dataset without 
    # considering class labels. On the contrary we treat different classes separately, we compute for 
    # them mean features, but for the covariance computation 
    # we normalize each sample's features using the mean of its class. By doing this we 
    # move all class clusters around the origin and build a single cluster
    # then we estimate the features covariance 

    # original code print results for different noise magnitudes. We inherit the magnitude from ODIN
    epsilon = 0.001

    train_lbls = train_loader.dataset.labels
    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    with torch.no_grad():
        example_input = train_loader.dataset[0][0].to(device).unsqueeze(0)
        n_known_classes = len(known_labels)

        # first we need to know the number of channels in each network layer
        # we obtain it by a fake forward
        feature_list = np.array([el.shape[1] for el in model_feature_list(model, example_input)[1]])

        print('Estimate sample mean and covariance from train data')
        sample_mean, precision = sample_estimator(model, n_known_classes, feature_list, train_loader, device)

    print('get Mahalanobis scores')

    magnitude = epsilon

    # for each test sample we compute a normality score for each output layer of the network
    for i in range(len(feature_list)):
        m_score, test_labels = get_Mahalanobis_score(
                model, 
                test_loader, 
                n_known_classes, 
                net_type="resnet",
                sample_mean=sample_mean, 
                precision=precision,
                layer_index=i, 
                magnitude=magnitude,
                device=device)
        m_score = np.asarray(m_score, dtype=np.float32)
        if i == 0:
            m_scores = m_score.reshape((m_score.shape[0], -1))
        else:
            m_scores = np.concatenate((m_scores, m_score.reshape((m_score.shape[0], -1))), axis=1)

    m_scores = np.asarray(m_scores, dtype=np.float32).T
    ood_labels = prepare_ood_labels(known_labels.numpy(), test_labels.numpy())

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")
    for l in range(len(m_scores)):
        metrics = calc_ood_metrics(m_scores[l], ood_labels)
        print(f"Layer {l} auroc: {metrics['auroc']:.4f}, fpr95: {metrics['fpr_at_95_tpr']:.4f}")

    return calc_ood_metrics(m_scores[-1], ood_labels)


# function to extract the multiple features
def model_feature_list(model, x):
    logits, feats = model(x)
    return logits, [feats]


def sample_estimator(model, num_classes, feature_list, train_loader, device):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of per-class mean
            precision: list of precisions
    """
    import sklearn.covariance
    covariance_estimator = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    # prepare some data structures
    correct, total = 0, 0
    num_output_layers = len(feature_list)

    num_sample_per_class = np.zeros((num_classes))
    list_features = [[0] * num_classes for _ in range(len(feature_list))]

    # for each model output layer, for each class, we extract for each sample 
    # the per-channel mean feature value
    for batch in tqdm(train_loader):

        data, target = batch
        total += data.size(0)
        data = data.to(device)
        output, out_features = model_feature_list(model, data)
        
        # compute the accuracy
        correct += (output.argmax(1).cpu() == target).sum()
        
        # construct the sample matrix
        for i in range(len(data)):
            label = target[i]
            # first sample for this class
            if num_sample_per_class[label] == 0:
                for layer_idx, layer_out in enumerate(out_features):
                    list_features[layer_idx][label] = layer_out[i].view(1,-1)
            else:
                for layer_idx, layer_out in enumerate(out_features):
                    list_features[layer_idx][label] = torch.cat((list_features[layer_idx][label], layer_out[i].view(1,-1)))
            num_sample_per_class[label] += 1
    
    sample_class_mean = []
    # for each output and for each class we compute the mean for each feat
    for layer_idx, layer_out in enumerate(feature_list):
        per_class_mean = torch.zeros((num_classes, layer_out)).to(device)
        for cls in range(num_classes):
            per_class_mean[cls] = torch.mean(list_features[layer_idx][cls], dim=0)
        sample_class_mean.append(per_class_mean)

    # we have computed features mean separately for each output layer and each class. 
    # now for each output layer we want to estimate the covariance matrix (and compute the inverse)
    # thus we move all samples of a lyer around the origin, by normalizing through their class mean
    # then we compute the covariance and its inverse
    precision = []
    for k in range(num_output_layers):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        covariance_estimator.fit(X.cpu().numpy())
        temp_precision = covariance_estimator.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('Training Accuracy:({:.2f}%)'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, net_type, sample_mean, precision, layer_index, magnitude, device):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index

    Similarly to what is done for ODIN, a first forward is computed, 
    then the predicted class for each sample is obtained
    (by looking at the mahalnobis distance from all classes),

    At this point a loss is computed using the predicted class as GT,
    the gradient on the input is used to apply a random noise on the input itself.
    This strategy should further separate known samples from unknown ones.

    The normality score is later computed using the corrupted input
    '''
    mahalanobis = []
    gt_labels = []
        
    for batch in tqdm(test_loader):
        data, target = batch
        gt_labels.append(target)
        
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        out_features = model(data)[1]
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)

        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # some strange magic on gradients 
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        # apply corruption
        tempInputs = torch.add(data, gradient, alpha=-magnitude)
        tempInputs = tempInputs.detach()
 
        # perform forward again 
        noise_out_features = model(tempInputs)[1]

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
    return mahalanobis, torch.cat(gt_labels)
