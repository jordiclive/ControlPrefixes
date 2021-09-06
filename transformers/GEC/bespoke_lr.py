import types
import warnings

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR



# class bespoke_LambdaLR(_LRScheduler):
#     """Sets the learning rate of each parameter group to the initial lr
#     times a given function. When last_epoch=-1, sets initial lr as lr.
#
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         lr_lambda (function or list): A function which computes a multiplicative
#             factor given an integer parameter epoch, or a list of such
#             functions, one for each group in optimizer.param_groups.
#         last_epoch (int): The index of last epoch. Default: -1.
#         verbose (bool): If ``True``, prints a message to stdout for
#             each update. Default: ``False``.
#
#     Example:
#         >>> # Assuming optimizer has two groups.
#         >>> lambda1 = lambda epoch: epoch // 30
#         >>> lambda2 = lambda epoch: 0.95 ** epoch
#         >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
#         >>> for epoch in range(100):
#         >>>     train(...)
#         >>>     validate(...)
#         >>>     scheduler.step()
#     """
#
#     def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
#         self.optimizer = optimizer
#
#         if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
#             self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
#         else:
#             if len(lr_lambda) != len(optimizer.param_groups):
#                 raise ValueError("Expected {} lr_lambdas, but got {}".format(
#                     len(optimizer.param_groups), len(lr_lambda)))
#             self.lr_lambdas = list(lr_lambda)
#         super(bespoke_LambdaLR, self).__init__(optimizer, last_epoch, verbose)
#
#     def state_dict(self):
#         """Returns the state of the scheduler as a :class:`dict`.
#
#         It contains an entry for every variable in self.__dict__ which
#         is not the optimizer.
#         The learning rate lambda functions will only be saved if they are callable objects
#         and not if they are functions or lambdas.
#
#         When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
#         """
#
#         state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
#         state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)
#
#         for idx, fn in enumerate(self.lr_lambdas):
#             if not isinstance(fn, types.FunctionType):
#                 state_dict['lr_lambdas'][idx] = fn.__dict__.copy()
#
#         return state_dict
#
#
#     def load_state_dict(self, state_dict):
#         """Loads the schedulers state.
#
#         When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
#
#         Args:
#             state_dict (dict): scheduler state. Should be an object returned
#                 from a call to :meth:`state_dict`.
#         """
#
#         lr_lambdas = state_dict.pop('lr_lambdas')
#         self.__dict__.update(state_dict)
#         # Restore state_dict keys in order to prevent side effects
#         # https://github.com/pytorch/pytorch/issues/32756
#         state_dict['lr_lambdas'] = lr_lambdas
#
#         for idx, fn in enumerate(lr_lambdas):
#             if fn is not None:
#                 self.lr_lambdas[idx].__dict__.update(fn)
#
#
#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.")
#         print('param_Groups', len(self.optimizer.param_groups[1]['params']))
#         print('param_Groups0', len(self.optimizer.param_groups[0]['params']))
#         print([base_lr * lmbda(self.last_epoch)
#                 for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)])
#         return [base_lr * lmbda(self.last_epoch)
#                 for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

def bespoke_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    def lr_lambda2(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps*3 - current_step) / float(max(1, num_training_steps*3 - num_warmup_steps))
        )

    return LambdaLR(optimizer, [lr_lambda,lr_lambda2], last_epoch)

