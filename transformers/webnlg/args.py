import sys
from collections import defaultdict
import argparse

import sys
from collections import defaultdict

# p = argparse.ArgumentParser()
# args, extras = p.parse_known_args()



import os, sys
import argparse
import time
if __name__ == '__main__':
    dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description = 'Args_save_checkpoint')
    parser.add_argument("--save_hf", type = str, default = '')
    parser.add_argument("--hf_checkpoint", type = bool, default = True)
    args, extras = parser.parse_known_args()
    print(args)
    print(extras)
    COMMANDLINE = f"python {dir}"+'/finetune_m_prefix.py '
    for i in vars(args).items():
        extras.append(i[0])
        extras.append(str(i[1]))
    COMMANDLINE += """ """.join(extras)
    print(COMMANDLINE)
    x = 2
    for i in vars(args).items():
        a.append(i[0])
        a.append(str(i[1]))
        # # Convert
        # COMMANDLINE = 'python {} ' \
        #               '--num_train_epochs {} ' \
        #               '--num_sanity_val_steps 1 ' \
        #               '--output_dir {} ' \
        #               '--data_dir {} ' \
        #               '--learning_rate {} ' \
        #               '--eval_min_length {} ' \
        #               '--gpus {} ' \
        #               '--eval_beams {} ' \
        #               '--precision {} ' \
        #               '--amp_level {} ' \
        #               '--val_metric {} ' \
        #               '--max_source_length={} ' \
        #               '--max_target_length={} ' \
        #               '--test_max_target_length={} ' \
        #               '--val_max_target_length={} ' \
        #               '--eval_max_gen_length={} ' \
        #               '--n_val=-1 ' \
        #               '--different_scheduler {} ' \
        #               '--new_tokens {} ' \
        #               '--save_hf {} ' \
        #               '--resume_from_checkpoint {} ' \
        #               '--hf_checkpoint {} ' \
        #               '--adafactor ' \
        #               '--preseqlen {}'.format(dir + '/finetune_m_prefix.py',
        #                                         args.num_train_epochs,
        #                                         args.output_dir,
        #                                         args.data_dir,
        #                                         args.learning_rate,
        #                                         args.eval_min_length,
        #                                         args.gpus,
        #                                         args.eval_beams,
        #                                         args.precision,
        #                                         args.amp_level,
        #                                         args.val_metric,
        #                                         args.max_source_length,
        #                                         args.max_target_length,
        #                                         args.test_max_target_length,
        #                                         args.val_max_target_length,
        #                                         args.eval_max_gen_length,
        #                                         args.different_scheduler,
        #                                         args.new_tokens,
        #                                      args.save_hf,
        #                                      args.resume_from_checkpoint,
        #                                      args.hf_checkpoint,
        #                                     args.preseqlen)
        #
        # print(COMMANDLINE)
        #
        # os.system(COMMANDLINE)

# if __name__ == '__main__':
#     dir = os.path.dirname(os.path.realpath(__file__))
#     parser = argparse.ArgumentParser(description='Args_save_checkpoint')
#     import sys
#     from collections import defaultdict
#     import argparse
#     args, extras = parser.parse_known_args()
#     print(args)
#     print(extras)
#     parser.add_argument("--save_hf", type = str, default = '')
#     parser.add_argument("--resume_from_checkpoint", type = str, default = '')
#     parser.add_argument("--hf_checkpoint", type = bool, default = True)
#
#
#     args = parser.parse_args()
#
#     # Convert
#     COMMANDLINE = 'python {} ' \
#                   '--num_train_epochs {} ' \
#                   '--num_sanity_val_steps 1 ' \
#                   '--output_dir {} ' \
#                   '--data_dir {} ' \
#                   '--learning_rate {} ' \
#                   '--eval_min_length {} ' \
#                   '--gpus {} ' \
#                   '--eval_beams {} ' \
#                   '--precision {} ' \
#                   '--amp_level {} ' \
#                   '--val_metric {} ' \
#                   '--max_source_length={} ' \
#                   '--max_target_length={} ' \
#                   '--test_max_target_length={} ' \
#                   '--val_max_target_length={} ' \
#                   '--eval_max_gen_length={} ' \
#                   '--n_val=-1 ' \
#                   '--different_scheduler {} ' \
#                   '--new_tokens {} ' \
#                   '--save_hf {} ' \
#                   '--resume_from_checkpoint {} ' \
#                   '--hf_checkpoint {} ' \
#                   '--adafactor ' \
#                   '--preseqlen {}'.format(dir + '/finetune_m_prefix.py',
#                                             args.num_train_epochs,
#                                             args.output_dir,
#                                             args.data_dir,
#                                             args.learning_rate,
#                                             args.eval_min_length,
#                                             args.gpus,
#                                             args.eval_beams,
#                                             args.precision,
#                                             args.amp_level,
#                                             args.val_metric,
#                                             args.max_source_length,
#                                             args.max_target_length,
#                                             args.test_max_target_length,
#                                             args.val_max_target_length,
#                                             args.eval_max_gen_length,
#                                             args.different_scheduler,
#                                             args.new_tokens,
#                                          args.save_hf,
#                                          args.resume_from_checkpoint,
#                                          args.hf_checkpoint,
#                                         args.preseqlen)
#
#     print(COMMANDLINE)
#
#     os.system(COMMANDLINE)
#
#     time.sleep(60)
#     os.system(f'rm -rf {args.output_dir}')
#
#     # Run Test
#     COMMANDLINE = 'python {} ' \
#                   '--num_train_epochs {} ' \
#                   '--num_sanity_val_steps 1 ' \
#                   '--output_dir {} ' \
#                   '--data_dir {} ' \
#                   '--learning_rate {} ' \
#                   '--eval_min_length {} ' \
#                   '--gpus {} ' \
#                   '--eval_beams {} ' \
#                   '--precision {} ' \
#                   '--amp_level {} ' \
#                   '--val_metric {} ' \
#                   '--max_source_length={} ' \
#                   '--max_target_length={} ' \
#                   '--test_max_target_length={} ' \
#                   '--val_max_target_length={} ' \
#                   '--eval_max_gen_length={} ' \
#                   '--n_val=-1 ' \
#                   '--different_scheduler {} ' \
#                   '--preseqlen {} ' \
#                   '--new_tokens {} ' \
#                   '--skip_train {} ' \
#                   '--prefixModel_name_or_path {} '.format(dir + '/finetune_m_prefix.py',
#                                             args.num_train_epochs,
#                                             args.output_dir,
#                                             args.data_dir,
#                                             args.learning_rate,
#                                             args.eval_min_length,
#                                             args.gpus,
#                                             args.eval_beams,
#                                             args.precision,
#                                             args.amp_level,
#                                             args.val_metric,
#                                             args.max_source_length,
#                                             args.max_target_length,
#                                             args.test_max_target_length,
#                                             args.val_max_target_length,
#                                             args.eval_max_gen_length,
#                                             args.different_scheduler,
#                                             args.preseqlen,
#                                             args.new_tokens,
#                                             args.skip_train,
#                                             args.save_hf + '/checkpoint-curr_best')
#
#
#     print(COMMANDLINE)
#
#     os.system(COMMANDLINE)
#

