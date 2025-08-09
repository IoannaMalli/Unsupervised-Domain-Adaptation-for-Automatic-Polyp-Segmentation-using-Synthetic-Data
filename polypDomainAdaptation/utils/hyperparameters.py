import copy

def init_ema_weights(model):

  ema_model = copy.deepcopy(model)
  ema_model.load_state_dict(model.state_dict())

  for parameters in ema_model.parameters():
    parameters.requires_grad = False

  return ema_model

# Updating during training
def update_ema_weights(segmentor,ema_model, alpha):
    for ema_param, model_param in zip(ema_model.parameters(), segmentor.parameters()):
        ema_param.data = alpha * ema_param.data + (1 - alpha) * model_param.data

def adjust_lambda_target(iterations, total_iterations=2500, lambda_initial=0.96, lambda_final=1):
    # Linear warm-up: early predictions are noisy, unsupervised loss unreliable
    # return lambda_initial + (lambda_final - lambda_initial) * ( iterations/ total_iterations)
    return 1

def adjust_alpha(iterations, total_iterations=2500, alpha_final=0.99):
    #  return min(1 - 1 / (iterations + 1), alpha_final)
    return 0.99

# def adjust_alpha(iterations, total_iterations=2500, alpha_initial=0.95, alpha_final=0.99):
#         return min(alpha_initial + (alpha_final - alpha_initial) * (iterations / total_iterations), alpha_final)

def adjust_pseudo_threshold(iterations, total_iterations=2500, thresh_init=0.9, thresh_final=0.968):
    # return thresh_init - (thresh_init - thresh_final) * (iterations / total_iterations)
    return 0.968
