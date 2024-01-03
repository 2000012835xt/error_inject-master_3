#!/usr/bin/env python3

"""
Main file to run the experiments.

Command:

    python3 main.py
"""
import pdb
import time

import torch
import torchvision
import torch.ao.quantization.quantize_fx as quantize_fx
import tqdm
from cifar10_task import Cifar10Task

import copy
import models
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import quant

def quantize(x, bit=8, scale=None, zero_point=0, all_positive=False, symmetric=False, per_channel=False):
        
    if all_positive:
        assert not symmetric, "Positive quantization cannot be symmetric"
        # unsigned activation is quantized to [0, 2^b-1]
        thd_neg = 0
        thd_pos = 2 ** bit - 1
    else:
        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            thd_neg = - 2 ** (bit - 1) + 1
            thd_pos = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            thd_neg = - 2 ** (bit - 1)
            thd_pos = 2 ** (bit - 1) - 1
    
    x = x / scale
    x = torch.clamp(x, thd_neg, thd_pos)
    xq = torch.round(x) + zero_point
    # x = (xq - zero_point) * scale 
    return xq


def fake_quant(x, bit=8, scale=None, zero_point=0, all_positive=False, symmetric=False, per_channel=False):
        
    if all_positive:
        assert not symmetric, "Positive quantization cannot be symmetric"
        # unsigned activation is quantized to [0, 2^b-1]
        thd_neg = 0
        thd_pos = 2 ** bit - 1
    else:
        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            thd_neg = - 2 ** (bit - 1) + 1
            thd_pos = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            thd_neg = - 2 ** (bit - 1)
            thd_pos = 2 ** (bit - 1) - 1
    
    x = x / scale
    x = torch.clamp(x, thd_neg, thd_pos)
    xq = torch.round(x) + zero_point
    x = (xq - zero_point) * scale 
    return x



def module_inject_error(model, module_names, prob, bw=32, bw_hardware=24):
    """
    prob: error probability, can be an integer or a list. When the prob is
    a list, prob[0] is the error probability for the LSB and prob[-1] is the
    error probability for the MSB.
    bw: hardware bitwith
    """

    def inject_error_hook(module_name, prob):
        print(f"Injecting error for {module_name}")

        """
        how should we inject error:
        - theoretically, error shows up during accumulation before the relu
          activation
        - error prob depends on the magnitude of psum
        - error prob depends on the bit location, e.g., LSB and MSB have
          different error prob --> 24b accum, 24 err prob

        conv (24b accum) -> relu (8 bit activation)
        """

        def inject_error(result, prob):
            # err_prob: result.shape + [bw]
            pdb.set_trace()
            err_prob = prob.repeat(list(result.shape) + [1])
            # generate error bit mask for each bit
            err_bit = torch.bernoulli(err_prob)
            weight = torch.Tensor([2 ** i for i in range(len(prob))])
            err = torch.sum(err_bit * weight, dim=-1, keepdim=False)
            err = err.type(result.dtype).to(result.device)

            # inject error into the integer part
            # 127 -> binary, err 15 -> binary --> xor
            # result, err: int32 format
            # 2's compliment for xor computation
            # result = torch.bitwise_xor(result, err)

            # sign magnitude
            # result = torch.where(
            #     result > 0,
            #     torch.bitwise_xor(result, err),
            #     - torch.bitwise_xor(torch.abs(result), err),
            # )

            # correct 2's compliment
            # when err < 0, 1100 --> 0100 - 2**(n - 1)
            err_comp = err + 2 ** (len(prob) - 1)
            result = torch.where(
                err >= 0,
                torch.bitwise_xor(result, err),
                torch.where(
                    result >= 0,
                    torch.bitwise_xor(result, err_comp) - 2 ** (bw_hardware - 1),
                    torch.bitwise_xor(result, err_comp) + 2 ** (bw_hardware - 1),
                )
            )
            return result

        """
        assume different bits are independent --> XOR

        forward_hook --> return 
        2's/SMR --> correct num (-128 - 127), incorrect num (-128 - 127)

        not each bit error prob for SMR format or 2C format

        eg: 010000 (16) -> 010010, (16 - 2 = 14) 001110

        eg: 010010 (18) -> 010000, (18 - 2 = 16) 010000
        """

        def inject_error_to_accum_hook(module, input, output):
            # pdb.set_trace()
            assert len(input) == 1
            # assert input[0].q_zero_point() == 0
            # assert module.weight().q_zero_point() == 0

            # qweight = module.weight().int_repr().type(torch.int32)
            # qinput = input[0].int_repr().type(torch.int32) - input[0].q_zero_point()
            # qresult = F.conv2d(qinput, qweight, None, module.stride, module.padding, module.dilation, module.groups)

            # result = F.conv2d(module.weight(), input[0], None, module.stride, module.padding, module.dilation, module.groups)
            # qresult = quantize(bit=8, scale=module.scale0 * module.scale1, all_positive=False)
            # pdb.set_trace()
            qweight = quantize(module.weight, scale=module.scale1)
            qinput = quantize(input[0], scale=module.scale0)
            qresult = F.conv2d(qinput, qweight, None, module.stride, module.padding)
            qresult = qresult.to(torch.int8)
            # pdb.set_trace()
            qresult = inject_error(qresult, prob)
            bias = module.bias[None, :, None, None]
            # output.int_repr() = qresult
            # qresult * scale + bias --> requant

            # result = torch.quantize_per_tensor(qresult * input[0].q_scale() * module.weight().q_scale() + bias,
            #                                    output.q_scale(), output.q_zero_point(), output.dtype)
            
            out = F.relu(qresult * module.scale0 * module.scale1 + bias)
            result = fake_quant(out, scale=module.scale2, all_positive=True)
            # 或者是 result = fake_quant(F.relu(qresult * module.scale0 * module.scale1 + bias), scale=module.scale2)
            

            # pdb.set_trace()
            # assert torch.allclose(result.int_repr(), output.int_repr(), 4), "result != output"
            return result

        # def inject_error_hook(module, input, output):
        #     # # err_prob: output.shape + [bw]
        #     # err_prob = prob.repeat(list(output.shape) + [1])
        #     # # generate error bit mask for each bit
        #     # err_bit = torch.bernoulli(err_prob)
        #     # weight = torch.Tensor([2 ** i for i in range(len(prob))])
        #     # err = torch.sum(err_bit * weight, dim=-1, keepdim=False)
        #     # err = err.type(output.int_repr().dtype).to(output.device)
        #     # # inject error into the integer part
        #     # output_err = torch.bitwise_xor(output.int_repr(), err)
        #     output_err = inject_error(output.int_repr(), prob)
        #     # dequantize
        #     output_err = (output_err - output.q_zero_point()) * output.q_scale()
        #     # requantize
        #     output_err = torch.quantize_per_tensor(output_err, output.q_scale(), output.q_zero_point(), torch.quint8)
        #     return output_err

        # return inject_error_hook
        return inject_error_to_accum_hook

    # pdb.set_trace()
    prob = (
        torch.Tensor([prob] * bw) if not isinstance(prob, list)
        else torch.Tensor(prob)
    )

    hooks = []
    print("Injection error rate is: ", prob[-1], prob[22], prob[21], prob[20])
    for name in module_names:
        module = get_module_by_name(model, name)
        hook = module.register_forward_hook(inject_error_hook(name, prob))
        hooks.append(hook)

    return hooks


def get_module_by_name(model, module_name):
    names = module_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def save_weight_to_memory(weight, csv_file_name):
    weight_np = []
    n, c, h, w = weight.size()
    for n_i in range(n):
        for h_i in range(h):
            for w_i in range(w):
                for c_i in range(c):
                    weight_np.append(weight[n_i][c_i][h_i][w_i].tolist())

    np.savetxt(csv_file_name, weight_np, fmt="%d", delimiter=',')
    return


def save_activation_to_memory(activation, csv_file_name):
    activation_np = []
    n, c, h, w = activation.size()
    for n_i in range(n):
        for h_i in range(h):
            for w_i in range(w):
                for c_i in range(c):
                    activation_np.append(activation[n_i][c_i][h_i][w_i].tolist())

    np.savetxt(csv_file_name, activation_np, fmt="%d", delimiter=',')
    return


def channel_sparsity_statis(activations):
    n, c, h, w = activations.size()
    zero_channel_count = 0
    for c_i in range(c):
        feature_sum = activations[0][c_i].sum()
        if feature_sum == 0:
            zero_channel_count = zero_channel_count + 1
    print("total channel: ", c)
    print("zero channel:", zero_channel_count)
    print("channel sparsity: ", 1 - zero_channel_count / c)


def print_module_inputs(model, module_names, inputs):
    """
    Collect the weights and activations for a given batch.
    The printed layers can be determined based on the model printed above
    """
    weights = {}
    activations = {}
    activations_out = {}

    def print_hook(module_name):
        print(f"Register print hook for {module_name}")

        def print_hook(module, input, output):
            assert len(input) == 1
            weights[module_name] = module.weight().clone().detach().cpu().int_repr()
            activations[module_name] = input[0].clone().detach().cpu().int_repr()
            activations_out[module_name] = output[0].clone().detach().cpu().int_repr()
            return output

        return print_hook

    for name in module_names:
        module = get_module_by_name(model, name)
        module.register_forward_hook(print_hook(name))

    model(inputs)
    return weights, activations, activations_out


def get_ptq_model(model, data_loader, num_batches, device):
    """
    Quantize the model with PTQ
    """
    model.eval()
    model.to(device)
    qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    ptq_model = quantize_fx.prepare_fx(model, {"": qconfig})
    for batch_idx, (inputs, _) in tqdm.tqdm(enumerate(data_loader)):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(device)
        ptq_model(inputs)
    return quantize_fx.convert_fx(ptq_model)


def get_qat_config(scheme):
    if scheme == "ternary_weight":
        w_quant = TernaryFakeQuantize(observer=MinMaxObserver,
                                      is_per_channel=True, thres=0.05)
        a_quant = NoopObserver.with_args(d_type=torch.float32)
    else:
        raise NotImplementedError
    return QConfig(activation=a_quant, weight=w_quant)


def train(model, optimizer, criterion, data_loader, device, epoch):
    model.train()
    loss = 0.
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, targets)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print_freq = 50
        if batch_idx % print_freq == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}/{len(data_loader)}",
                'loss: %.3f | acc: %.3f%% (%d/%d)' % (
                    loss / (batch_idx + 1),
                    100. * correct / total,
                    correct,
                    total
                )
            )

    return loss, float(correct / total)


def test(model, criterion, data_loader, device, epoch):
    """
    Evaluate the loss and accuracy of the model
    """

    model.eval()
    correct = 0
    total = 0
    loss = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)

            loss += batch_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print_freq = 20
            if batch_idx % print_freq == 0:
                print(
                    f"Epoch {epoch} Batch {batch_idx}/{len(data_loader)}",
                    'loss: %.3f | acc: %.3f%% (%d/%d)' % (
                        loss / (batch_idx + 1),
                        100. * correct / total,
                        correct,
                        total
                    )
                )
    print(f"Top1 accuracy: {100. * correct / total}%")
    return loss, float(correct / total)


def test_one_time(model, criterion, data_loader, device, epoch):
    """
    Evaluate the loss and accuracy of the model
    """

    model.eval()
    correct = 0
    total = 0
    loss = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx == 0:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)

                loss += batch_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print_freq = 20
                if batch_idx % print_freq == 0:
                    print(
                        f"Epoch {epoch} One Batch/{len(data_loader)}",
                        'loss: %.3f | acc: %.3f%% (%d/%d)' % (
                            loss / (batch_idx + 1),
                            100. * correct / total,
                            correct,
                            total
                        )
                    )
            break

    # print(f"Top1 accuracy: {100. * correct / total}%")
    return loss, float(correct / total)



def build_optimizer(model, lr):
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def build_scheduler(optimizer, max_epoch):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


def save_checkpoint(model, filename):
    if os.path.exists(filename):
        raise ValueError
    else:
        torch.save({"model_state_dict": model.cpu().state_dict()}, filename)


def load_checkpoint(model, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def read_error_rate(file, start_row):
    error_rate = []
    with open(file, 'r') as fr_error_rate:
        line = fr_error_rate.readline()
        index_bit_err = 0
        while line:
            if index_bit_err >= start_row:
                line_data = line.split(',')
                error_rate_bit_wise = []
                for error in line_data:
                    if error == '\n':
                        continue
                    error_rate_bit_wise.append(eval(error) / 2)  # meta stable model random to 0 or 1
                    # error_rate_bit_wise.append(eval(error))       # bit flip model
                error_rate.append(error_rate_bit_wise)
            index_bit_err = index_bit_err + 1
            line = fr_error_rate.readline()
    return error_rate


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # device = "cpu"
    print("the device used is ", device)

    # save_file = './models/lenet_cifar.pkl'
    # save_file = './models/alexnet_cifar.pkl'

    # save_file = './models/resnet18_s8.pt'
    save_file = "./models/vgg16_bn.pt"

    # save_file = './models/mobilenetv1_s8.pt'
    # save_file = './models/mobilenetv1_0p5_s8.pt'

    # create criterion, training and test data loader
    time_start = time.time()
    task = Cifar10Task(data_root='/home/zzd/zzd_data/PycharmProjects/data')
    criterion = task.get_criterion().to(device)
    train_loader = task.get_train_dataloader(batch_size=128)
    test_loader = task.get_test_dataloader(batch_size=128)

    # load pre-trained model
    # model = models.LeNet()
    # model = models.AlexNet()
    # model.load_state_dict(torch.load(save_file))

    # model = models.resnet18(pretrained=False, device=device)
    model = models.vgg16_bn(num_classes=10)

    # pdb.set_trace()

    # model = models.mobilenetv1(num_classes=10)
    # model = models.mobilenetv1_0p5(num_classes=10)
    load_checkpoint(model, save_file)
    # model.to("cpu")

    # evaluate the accuracy of the pre-trained model
    # test(model, criterion, test_loader, device, epoch=0)

    ####  load the trained model
    # test(model, criterion, test_loader, device, epoch=0)

    # quantize model with ptq and evaluate accuracy
    '''
    NOTE: quantized model can only run on cpu due to kernel availability issue
    但这样速度太慢, 自己创建fake quantized model, load ptq model参数, 就可以用GPU跑, 实现加速
    '''
    ptq_model = get_ptq_model(model, train_loader, num_batches=128, device=device)
    print(ptq_model)

    # for name, module in ptq_model.named_modules(): 
    #     print(name, module)

    # pdb.set_trace()

    ptq_model = ptq_model.to('cpu').eval()

    conv_weights = []
    conv_bias = []
    conv_w_scale = []
    conv_a_scale = []
    input_scale = []
    input_zero_point = []

    def save_conv_data(module, input, output):
        conv_weights.append(module.weight().dequantize())
        conv_bias.append(module.bias().dequantize())
        conv_w_scale.append(module.weight().q_scale())
        # conv_a_scale.append(input[0].q_scale())
        conv_a_scale.append(output.q_scale())
        input_scale.append(input[0].q_scale())
        input_zero_point.append(input[0].q_zero_point())
        # pdb.set_trace()


    
    linear_weights = []
    linear_bias = []
    linear_w_scale = []
    linear_a_scale = []
    linear_zero_point = []

    def save_linear_data(module, input, output):
        linear_weights.append(module.weight().dequantize())
        linear_bias.append(module.bias().dequantize())
        linear_w_scale.append(module.weight().q_scale())
        # linear_a_scale.append(input[0].q_scale())
        linear_a_scale.append(output.q_scale())
        linear_zero_point.append(output.q_zero_point())


    #把ptq_model的weight, bias, scale通过hook保存出来，然后写到fake quant model里
    for name, module in ptq_model.named_modules(): 
        if isinstance(module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d):
            module.register_forward_hook(save_conv_data)
        if isinstance(module, torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU) or\
            isinstance(module, torch.nn.quantized.modules.linear.Linear):
            module.register_forward_hook(save_linear_data)
            
    # test(ptq_model, criterion, test_loader, 'cpu', epoch=0)
    test_one_time(ptq_model, criterion, test_loader, 'cpu', epoch=0) #只测试一个batch, 用于得到数据

    fake_quant_model = quant.quant_VGG16(10, input_scale[0], input_zero_point[0], conv_weights, conv_bias, 
                                         linear_weights, linear_bias, input_scale,
                                         conv_w_scale, conv_a_scale, linear_w_scale, linear_a_scale, linear_zero_point[-1])


    test(fake_quant_model, criterion, test_loader, device, epoch=0)

    # pdb.set_trace()

    # ptq_model.to(device)

    # inject error to module in alexnet and resnet
    # # 31 - 0 -> 31 and 22 - 0
    # error_rate_file = ['./error_rate/alexnet_cnn2.csv',
    #                    './error_rate/alexnet_cnn3.csv', './error_rate/alexnet_cnn4.csv']
    # layer_injection = ["cnn2", "cnn3", "cnn4"]

    # error_rate_file = ['./error_rate/alexnet_test.csv', './error_rate/alexnet_test.csv',
    #                    './error_rate/alexnet_test.csv', './error_rate/alexnet_test.csv']
    # layer_injection = ["cnn1", "cnn2", "cnn3", "cnn4"]
    # log_file = "./log/alexnet_accuracy.csv"

    # error_rate_file = ['./error_rate/resnet/layer3.csv', './error_rate/resnet/layer5.csv',
    #                    './error_rate/resnet/layer7.csv', './error_rate/resnet/layer9.csv',
    #                    './error_rate/resnet/layer11.csv', './error_rate/resnet/layer13.csv',
    #                    './error_rate/resnet/layer15.csv', './error_rate/resnet/layer17.csv']
    # layer_injection = ["layer1.1.conv1", "layer2.0.conv1", "layer2.1.conv1", "layer3.0.conv1", "layer3.1.conv1",
    #                    "layer4.0.conv1", "layer4.1.conv1", "layer4.1.conv2"]
    # log_file = "./log/resnet_accuracy.csv"

    error_rate_file = ['./error_rate/vgg/feature6.csv', './error_rate/vgg/feature9.csv',
                       './error_rate/vgg/feature13.csv', './error_rate/vgg/feature19.csv',
                       './error_rate/vgg/feature23.csv', './error_rate/vgg/feature26.csv',
                       './error_rate/vgg/feature33.csv', './error_rate/vgg/feature39.csv']
    
    # layer_injection = ["features.6", "features.9", "features.13", "features.19",
    #                    "features.23", "features.26", "features.33", "features.39"]
    
    layer_injection = ["features.2", "features.3", "features.5", "features.7",
                       "features.9", "features.10", "features.13", "features.15"]
    
    log_file = "./log/vgg_accuracy.csv"

    error_rate_all_layer = []
    for error_rate_file_i in error_rate_file:
        error_rate_all_layer.append(read_error_rate(error_rate_file_i, 1))

    accuracy = []
    for i in range(len(error_rate_all_layer[0])):
        error_prob_injection = []
        for layer_i, error_rate_all_layer_i in enumerate(error_rate_all_layer):
            error_prob_current_layer = [0] * 32
            error_prob_current_layer[31] = error_rate_all_layer_i[i][0]
            for bit_i, error_bit_i in enumerate(error_rate_all_layer_i[i]):
                if bit_i > 0:
                    error_prob_current_layer[23 - bit_i] = error_bit_i
            error_prob_injection.append(error_prob_current_layer)

        acc_one_er = []
        for repeat_i in range(2):
            # ptq_model_cpy = copy.deepcopy(ptq_model)
            # fake_quant_model_cpy = copy.deepcopy(fake_quant_model)

            time_inject_start = time.time()
            torch.manual_seed(repeat_i * 13)

            hooks = []

            for layer_i, layer in enumerate(layer_injection):
                # module_inject_error(ptq_model_cpy, [layer], prob=error_prob_injection[layer_i], bw=32,
                #                     bw_hardware=24)
                layer_hooks = module_inject_error(fake_quant_model, [layer], prob=error_prob_injection[layer_i], bw=32,
                                    bw_hardware=24)
                hooks.extend(layer_hooks)
            # module_inject_error(ptq_model_cpy, ["cnn0"], prob=0, bw=32, bw_hardware=24)
            # ptq_model_cpy(next(iter(test_loader))[0])
            # loss, acc = test(ptq_model_cpy, criterion, test_loader, device, epoch=0)
            loss, acc = test(fake_quant_model, criterion, test_loader, device, epoch=0)

            time_inject_end = time.time()
            print("injection run time: ", time_inject_end - time_inject_start)
            acc_one_er.append(acc)

            # clear hooks
            for hook in hooks:
                hook.remove()

        accuracy.append(acc_one_er)
    np.savetxt(log_file, accuracy, fmt="%.4e", delimiter=',')

    time_end = time.time()
    print("total runtime: ", time_end - time_start)

    # inject error to module in lenet
    # # 31 - 0 -> 31 and 22 - 0
    # if save_file == './models/lenet_cifar.pkl':
    #     error_rate_file = './error_rate/lenet_cnn1_fate.csv'
    #     error_rate_all = read_error_rate(error_rate_file, 1)
    #     accuracy = []
    #     for i in range(len(error_rate_all)):
    #         error_prob = [0] * 32
    #         error_prob[31] = error_rate_all[i][0]
    #         for j in range(len(error_rate_all[i])):
    #             if j > 0:
    #                 error_prob[23 - j] = error_rate_all[i][j]
    #         acc_one_er = []
    #         for repeat_i in range(5):
    #             ptq_model_cpy = copy.deepcopy(ptq_model)
    #             time_inject_start = time.time()
    #             torch.manual_seed(repeat_i * 3)
    #             module_inject_error(ptq_model_cpy, ["cnn1"], prob=error_prob, bw=32, bw_hardware=24)
    #             # module_inject_error(ptq_model_cpy, ["cnn0"], prob=0, bw=32, bw_hardware=24)
    #             # ptq_model_cpy(next(iter(test_loader))[0])
    #             loss, acc = test(ptq_model_cpy, criterion, test_loader, "cpu", epoch=0)
    #             time_inject_end = time.time()
    #             print("injection run time: ", time_inject_end - time_inject_start)
    #             acc_one_er.append(acc)
    #         accuracy.append(acc_one_er)
    #     np.savetxt("./accuracy.csv", accuracy, fmt="%.4e", delimiter=',')
