# 모델 얼리는 코드
def freez_layer(layer):
    for p in layer.parameters():
            p.requires_grad = False

def freez_model(model_name, model, freez_layers=[]):
    # print(model)
    if model_name == 'squeezenet' or model_name == 'sky_squeezenet':
        for i, layer in enumerate(model.features):
            if i in [0, 3, 4, 6, 7]:
                freez_layer(layer)
    if model_name == 'squeezenet1.1':
        for i, layer in enumerate(model.features):
            if i < 1 or i > 11: continue
            freez_layer(layer)
    elif model_name == 'squeezenet1.1_small_freeze':
        for i, layer in enumerate(model.features):
            if i < 3 or i > 9: continue
            freez_layer(layer)
    elif model_name == 'resnet18':
        print('resnet18 freez layer1~3')
        freez_layer(model.layer1)
        freez_layer(model.layer2)
        freez_layer(model.layer3)
        # freez_layer(model.layer4)
    elif model_name == 'resnet50':
        for i in freez_layers:
            if i == 1: freez_layer(model.layer1)
            if i == 2: freez_layer(model.layer2)
            if i == 3: freez_layer(model.layer3)
            if i == 4: freez_layer(model.layer4)
            print('freeze layer : ', i)
        # print('resnet50 freez layer 1~2')
        # freez_layer(model.layer1)
        # freez_layer(model.layer2)
        # freez_layer(model.layer3)
        # freez_layer(model.layer4)
    elif model_name == 'densnet':
        print('densnet freez layer1~3')
        freez_layer(model.layer1)
        freez_layer(model.layer2)
        freez_layer(model.layer3)
        # freez_layer(model.layer4)


def add_weight_decay(models, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in models.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay':0.}, {'params':decay, 'weight_decay': l2_value}]


