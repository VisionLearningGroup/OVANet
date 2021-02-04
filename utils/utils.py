from models.basenet import *
import os
import torch
import neptune
import socket


def get_model_mme(net, num_class=13, temp=0.05, top=False, norm=True):
    dim = 2048
    if "se_" in net:
        model_g = SEBase(net, top=top)
        if net == "se_resnet18":
            dim = 512
        if net == "se_resnet34":
            dim = 512
    elif "shuffle" in net:
        model_g = ShuffleBase(net, top=top)
        dim = 1024
    elif "inception" in net:
        model_g = IncepionBase(net, top=top)
        dim = 1024
    elif "efficient" in net:
        model_g = EfficientBase(net, top=top)
        dim = 1280
        if "b4" in net:
            dim = 1792
    elif "wide_resnet" in net:
        model_g = WideRes(net, top=top)
    elif "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "resnext" in net:
        model_g = ResNextBase(net, top=top)
    elif "densenet" in net:
        model_g = DenseBase(net, top=top)
        dim = 1024
        if "161" in net:
            dim = 2208
        if "169" in net:
            dim = 1664
        if "201" in net:
            dim = 1920
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    elif "alex" in net:
        model_g = models.alexnet(pretrained=True)
        print('alex')
        dim = 1000
    if top:
        dim = 1000
    print("selected network %s"%net)
    model_c = ResClassifier_MME(num_classes=num_class, temp=temp, input_size=dim, norm=norm)
    return model_g, model_c, dim

def log_set(source_data, target_data, network, args,
            script_name, conf_file, gpu_devices):
    logname = "{file}_{source}2{target}_{setting}_{network}_hp_{hp}".format(file=script_name.replace(".py", ""),
                                                                               source=source_data.split("_")[1],
                                                                               target=target_data.split("_")[1],
                                                                               setting=source_data.split("_")[
                                                                                   -1].replace(".txt", ""),
                                                                               network=network,
                                                                               hp=str(args.multi))
    logname = os.path.join("record", args.exp_name,
                           os.path.basename(conf_file).replace(".yaml", ""), logname)
    if args.neptune:
        neptune.init('keisaito/sandbox')
        PARAMS = {'learning rate': args.multi,
                  'machine': socket.gethostname(),
                  'gpu': gpu_devices[0],
                  'file': script_name,
                  'net': network}
        neptune.create_experiment(name=logname, params=PARAMS)
        neptune.append_tag("config %s file %s" % (conf_file, logname))
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))
    print("record in %s " % logname)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logname, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("{}_2_{}".format(source_data, target_data))
    return logname


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c

def normed_kl(p, p_proxy):
    kl = torch.sum(p_proxy * torch.log(p_proxy / p +1e-10), 1)
    max_kl = - math.log(1. / p_proxy.size(1) +1e-10)
    return kl / max_kl

def kl_sym(p1, p2):
    p1 = F.softmax(p1) + 1e-10
    p2 = F.softmax(p2) + 1e-10
    kl = torch.sum(p1 * torch.log(p1 / p2 +1e-10), 1)
    kl += torch.sum(p2 * torch.log(p2 / p1 +1e-10), 1)
    return torch.mean(kl) * 0.5
