def freeze_prefix1(model):
    base_prefix = []
    params = [p for n, p in model.named_parameters() if not any(nd in n for nd in ["CEFR_matrices"])]
    for par in params:
        par.requires_grad = False

def transfer_prefix1(model,model_base):

    params = model.named_parameters()
    params_base_ = model_base.named_parameters()
    dict_params = dict(params)
    params_base = [(n,p) for n, p in model_base.named_parameters() if not any(nd in n for nd in base_prefix)]


    for name_base, param_base in params_base:
            dict_params[name_base].data.copy_(param_base.data)

    model_base.load_state_dict(dict_params)





