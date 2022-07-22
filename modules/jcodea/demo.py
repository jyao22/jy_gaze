#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# The code from 'modules' directory is mounted under /project/modules, which is set as env PYTHONPATH.
# Therefore, you could import modules from this folder as is.
# 
# For more info (other mounting paths, parameters, etc), refer to [neuro-flow jupyter action repository](https://github.com/neuro-actions/jupyter).

from train import train, get_parser

arg_parser = get_parser()
args = arg_parser.parse_args(["--data_dir", "/project/data"])

train(args)

