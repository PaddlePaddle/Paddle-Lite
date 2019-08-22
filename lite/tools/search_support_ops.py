# -*- coding: utf-8 -*- 
import os
import re


def merge_sort_two_list(la, lb):
    la.extend(lb)
    la = list(set(la))
    la.sort()
    return la


ops_file = "../api/paddle_use_ops.h"
kernels_file = "../api/paddle_use_kernels.h"
result_file = "./support_ops_list.md"

# search ops
if os.path.exists(ops_file):
    pattern = re.compile("USE_LITE_OP[(](.*?)[)]")
    ops = []
    for line in open(ops_file):
        if line != None and line[0:2] != "//":
            op = pattern.findall(line)
            ops.extend(op)
    ops.sort()
    # print ops
    # print len(ops)
else:
    print "ops_file no exist in ", ops_file

# search kernels
if os.path.exists(kernels_file):
    kernel_types = [
        "kARM, kFloat", "kARM, kInt8", "kARM, kAny", "kX86, kFloat",
        "kX86, kInt8", "kX86, kAny", "kOpenCL, kFloat", "kOpenCL, kInt8",
        "kOpenCL, kAny"
    ]
    patterns = []
    for type in kernel_types:
        pat_str = "USE_LITE_KERNEL[(](.*?), " + type
        patterns.append(re.compile(pat_str))

    kernels = [[] for i in range(len(kernel_types))]
    for line in open(kernels_file):
        if line != None and line[0:2] != "//":
            for i in range(len(kernel_types)):
                kl = patterns[i].findall(line)
                kernels[i].extend(kl)
else:
    print "kernels_file no exist in ", kernels_file

# write out
if os.path.exists(result_file):
    os.remove(result_file)
out = open(result_file, "w")
out.write("# PaddleLite support ops and kernels\n")
out.write("## ops\n")
for op in ops:
    out.write("- " + op + "\n")

out.write("## kernels\n")
for i in range(len(kernel_types) / 3):
    for j in range(2):
        out.write("### " + kernel_types[3 * i + j] + "\n")
        for kl in merge_sort_two_list(kernels[3 * i + j], kernels[3 * i + 2]):
            out.write("- " + kl + "\n")
