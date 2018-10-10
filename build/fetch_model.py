from src.model.llp_lr import LabelRegularization


def fetch(model_name, config, args):
    if model_name == 'llp_lr':
        cls = LabelRegularization(config, args)
        return cls
    else:
        return None


def buildiup(self):
    model = None
    return model
