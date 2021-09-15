#!/usr/bin/env python
# Copyright @ 2020 Baidu. All rights reserved.
""" python wrapper file for Paddle-Lite opt tool """
from __future__ import print_function
import paddlelite.lite as lite
import argparse
import subprocess

def main():
    """ main funcion """
    a=lite.Opt()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=False,\
        help="path of the model. This option will be ignored if model_file and param_file exist")
    parser.add_argument("--model_file", type=str, required=False,\
        help="model file path of the combined-param model.")
    parser.add_argument("--param_file", type=str, required=False,\
        help="param file path of the combined-param model.")
    parser.add_argument("--optimize_out_type", type=str, required=False,default="naive_buffer",\
        choices=['protobuf', 'naive_buffer'], \
        help="store type of the output optimized model. protobuf/naive_buffer.")
    parser.add_argument("--optimize_out", type=str, required=False,\
        help="path of the output optimized model")
    parser.add_argument("--valid_targets", type=str, required=False,default="arm",\
        help="The targets this model optimized for, should be one of (arm,opencl, x86), splitted by space.")

    # arguments of quantization
    parser.add_argument("--quant_model", type=str, default="false",
        help="{true, false} Use post_quant_dynamic method to quantize"
             "the model weights. Default false.")
    parser.add_argument("--quant_type", type=str, default="QUANT_INT16",
        help="{QUANT_INT16, QUANT_INT8} Set the quant_type for "
             "post_quant_dynamic. Default QUANT_INT16.")
    parser.add_argument("--enable_fp16", type=str, default="false",
        help="{true, false} Whether to enable FP16 calculation, FP16 "
             "calculation will cause a lower precision but higher inference speed.")

    # arguments of sparsification
    parser.add_argument("--sparse_model", type=str, default="false",
        help="{true, false} Use sparse_conv_detect_pass sparsify"
             "the 1x1conv weights. Default false.")
    parser.add_argument("--sparse_threshold", type=float, default=0.6,\
        help="Set a value to determine the lower bound for sparse pass. \
              0.6 means sparse pass will be skipped if the current weight sparsity is smaller than 0.6.")

   # arguments of help information
    parser.add_argument('--version', action='version', version=a.version())
    parser.add_argument("--print_supported_ops", type=str, default="false",\
        help="{true, false}\
               Print supported operators on the inputed target")
    parser.add_argument("--print_all_ops", type=str, default="false",\
        help="{true, false}\
               Print all the valid operators of Paddle-Lite")
    parser.add_argument("--print_all_ops_in_md_format", type=str, default="false",\
        help="{true, false}\
               Print all the valid operators of Paddle-Lite in markdown format")
    parser.add_argument("--print_model_ops", type=str, default="false",\
        help="{true, false}\
               Print operators in the input model")
    parser.add_argument("--display_kernels", type=str, default="false",\
        help="{true, false}\
               Display kernel information")

   # arguments of strip lib according to input model
    parser.add_argument("--record_tailoring_info", type=str, default="false",\
        help="{true, false}\
               Record kernels and operators information of the optimized model \
               for tailoring compiling, information are stored into optimized  \
               model path as hidden files")
    parser.add_argument("--model_set", type=str, required=False,\
        help="path of the models set. This option will be used to specific \
              tailoring")

    # arguments of visualization optimized nb model
    parser.add_argument("--optimized_nb_model_path", type=str, required=False,\
        help="path of the optimized nb model, this argument is use for the VisualizeOptimizedModel API")
    parser.add_argument("--visualization_file_output_path", type=str, required=False,\
        help="output path of the visualization file, this argument is use for the VisualizeOptimizedModel API")

    args = parser.parse_args()
    """ input opt params """
    if args.model_dir is not None:
         a.set_model_dir(args.model_dir)
    if args.model_set is not None:
         a.set_modelset_dir(args.model_set)
    if args.model_file is not None:
         a.set_model_file(args.model_file)
    if args.param_file is not None:
         a.set_param_file(args.param_file)
    if args.optimize_out_type is not None:
         a.set_model_type(args.optimize_out_type)
    if args.optimize_out is not None:
         a.set_optimize_out(args.optimize_out)
    if args.valid_targets is not None:
         if args.enable_fp16 == "true":
              a.enable_fp16()
              a.set_valid_places(args.valid_targets)
         else:
              a.set_valid_places(args.valid_targets)
    if args.param_file is not None:
         a.set_param_file(args.param_file)
    if args.record_tailoring_info == "true":
         a.record_model_info(True)
    if args.quant_model == "true":
        a.set_quant_model(True)
        a.set_quant_type(args.quant_type)
    if args.sparse_model == "true":
        a.set_sparse_model(True)
        a.set_sparse_threshold(args.sparse_threshold)
    """ print ops info """
    if args.print_all_ops == "true":
         a.print_all_ops()
         return 0
    if args.print_all_ops_in_md_format == "true":
         a.print_all_ops_in_md_dormat()
         return 0
    if args.print_supported_ops == "true":
         a.print_supported_ops()
         return 0
    if args.display_kernels == "true":
         a.display_kernels_info()
         return 0
    if args.print_model_ops == "true":
         a.check_if_model_supported(True);
         return 0
    """ visualize optimized naive buffer model """
    if args.optimized_nb_model_path is not None and args.visualization_file_output_path is not None:
        file_names = a.visualize_optimized_nb_model(args.optimized_nb_model_path, args.visualization_file_output_path)
        for file_name in file_names:
          dot_path = args.visualization_file_output_path + file_name + ".dot"
          pdf_path = args.visualization_file_output_path + file_name + ".pdf"
          cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
          subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 0
    if ((args.model_dir is None) and (args.model_file is None or args.param_file is None) and (args.model_set is None)) or (args.optimize_out is None):
         a.executablebin_help()
         return 1
    else:
         a.run()
         return 0
if __name__ == "__main__":
    main()
