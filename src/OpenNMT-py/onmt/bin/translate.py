#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
from onmt import opts
from onmt.utils.parse import ArgumentParser
from onmt.train_single import load_dataset

def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator, model_opt = build_translator(opt, report_score=True)
    test_data = load_dataset(model_opt.task_name, model_opt.architect, opt.src, opt.tgt)

    logger.info("Translating test set %s." % opt.src)
    _, all_preds = translator.translate(
        data = test_data,
        batch_size=opt.batch_size,
    )
    if opt.output is not None:
        with open(opt.output, 'w') as f:
            for pred in all_preds:
                f.write(pred)
                f.write("\n")


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
