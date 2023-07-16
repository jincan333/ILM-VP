import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import types


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(masks, net, keep_ratio, train_dataloader, device, visual_prompt, label_mapping):

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad = True
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net).to(device)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    # outputs = label_mapping(net.forward(visual_prompt(inputs))) if visual_prompt else label_mapping(net.forward(inputs))
    outputs = net.forward(visual_prompt(inputs)) if visual_prompt else net.forward(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    grads_abs={}
    for name, weight in net.named_parameters():
        if name[:-5] not in masks: continue
        grads_abs[name[:-5]] = torch.abs(weight.grad)

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs.values()])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    # calculate mask
    for name in grads_abs.keys():
        def unravel_index(index, shape):
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))

        for name in grads_abs.keys():
            mask = ((grads_abs[name] / norm_factor) >= acceptable_score).float()
            if torch.sum(mask) == 0:  # If all values are 0...
                flat_idx = torch.argmax(grads_abs[name])
                unflat_idx = unravel_index(flat_idx, grads_abs[name].size())
                mask[unflat_idx] = 1  # set the maximum argument to 1

            masks[name] = mask


    # for name in grads_abs.keys():
        # masks[name] = ((grads_abs[name] / norm_factor) >= acceptable_score).float()


def SNIP_training(net, keep_ratio, train_dataloader, device, masks, death_rate):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    print('Pruning rate:', death_rate)
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    # for layer in net.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
    #         # nn.init.xavier_normal_(layer.weight)
    #         # layer.weight.requires_grad = False
    #
    #     # Override the forward methods:
    #     if isinstance(layer, nn.Conv2d):
    #         layer.forward = types.MethodType(snip_forward_conv2d, layer)
    #
    #     if isinstance(layer, nn.Linear):
    #         layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = []
    masks_copy = []
    new_masks = []
    for name in masks:
        masks_copy.append(masks[name])

    index = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # clone mask
            mask = masks_copy[index].clone()

            num_nonzero = (masks_copy[index] != 0).sum().item()
            num_zero = (masks_copy[index] == 0).sum().item()

            # calculate score
            scores = torch.abs(layer.weight.grad * layer.weight * masks_copy[index]) # weight * grad
            norm_factor = torch.sum(scores)
            scores.div_(norm_factor)

            x, idx = torch.sort(scores.data.view(-1))
            num_remove = math.ceil(death_rate * num_nonzero)
            k = math.ceil(num_zero + num_remove)
            if num_remove == 0.0: return masks_copy[index] != 0.0

            mask.data.view(-1)[idx[:k]] = 0.0

            new_masks.append(mask)
            index += 1

    return new_masks


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    samples_per_class = min(int(512 // num_classes), samples_per_class)
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        try:
            inputs, targets = next(dataloader_iter)
        except StopIteration:
            print("The iterator has now run out of items.")
            break
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def GraSP(masks, net, ratio, train_dataloader, device, visual_prompt, label_mapping, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = ratio
    old_net = net
    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    inputs_one = []
    targets_one = []
    grad_w = None
    for w in weights:
        w.requires_grad_(True)
    print_once = False
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)

        # outputs = label_mapping(net.forward(visual_prompt(inputs[:N//2])))/T if visual_prompt else label_mapping(net.forward(inputs[:N//2]))/T
        outputs = net.forward(visual_prompt(inputs[:N//2]))/T if visual_prompt else net.forward(inputs[:N//2])/T
        if print_once:
            # import pdb; pdb.set_trace()
            x = F.softmax(outputs)
            print(x)
            print(x.max(), x.min())
            print_once = False
        loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        # outputs = label_mapping(net.forward(visual_prompt(inputs[N // 2:])))/T if visual_prompt else label_mapping(net.forward(inputs[N // 2:]))/T
        outputs = net.forward(visual_prompt(inputs[N // 2:]))/T if visual_prompt else net.forward(inputs[N // 2:])/T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        # outputs = label_mapping(net.forward(visual_prompt(inputs)))/T  if visual_prompt else label_mapping(net.forward(inputs))/T
        outputs = net.forward(visual_prompt(inputs))/T  if visual_prompt else net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    for name, weight in net.named_parameters():
        if name not in masks: continue
        grads[name] = -weight.data * weight.grad

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    # calculate mask
    for name in grads.keys():
        masks[name] = ((grads[name] / norm_factor) <= acceptable_score).float()



def SynFlow(masks, net, keep_ratio, train_dataloader, device, visual_prompt, label_mapping):

    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad = True

    net = copy.deepcopy(net).to(device)
    scores = {}
    epochs=100
    # mask iteratively
    for epoch in range(epochs):
        # linearize
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        # input_dim = list(inputs[0, :].shape)
        # input = torch.ones([1] + input_dim).to(device)
        # Use visual prompt
        input = inputs
        output = net(visual_prompt(input)) if visual_prompt else net(input)
        torch.sum(output).backward()
        # calculate scores
        for name, weight in net.named_parameters():
            if name not in masks: continue
            scores[name] = torch.clone(weight.grad * weight).detach().abs_()
            weight.grad.data.zero_()
        # unlinearize
        for name, param in net.state_dict().items():
            param.mul_(signs[name])
        # calculate ratio to mask
        ratio = keep_ratio**((epoch+1)/epochs)
        # calculate mask
        global_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        k = int((1 - ratio) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for name, weight in net.named_parameters():
                if name not in masks: continue
                score = scores[name] 
                zero = torch.tensor([0.]).to(device)
                one = torch.tensor([1.]).to(device)
                masks[name].copy_(torch.where(score <= threshold, zero, one))
        # apply mask to net
        for name, weight in net.named_parameters():
            if name not in masks: continue
            weight.data=weight.data.mul_(masks[name])
