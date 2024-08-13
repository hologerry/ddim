from torch.optim import SGD, Adam, RMSprop


def get_optimizer(config, parameters):
    if config.optim.optimizer == "Adam":
        return Adam(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )
    elif config.optim.optimizer == "RMSProp":
        return RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == "SGD":
        return SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not understood.")
