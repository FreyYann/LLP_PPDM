from src.model.llp_lr import LabelRegularization


def fetch(model_name, config):
    if model_name == 'llp_lr':
        cls = LabelRegularization(config)
        return cls
    else:
        return None


def buildiup(self):
    model = None
    return model
