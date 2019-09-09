def gen_use_kernel_statement(op_type, target, precision, layout, alias):
    return 'USE_LITE_KERNEL(%s, %s, %s, %s, %s);' %(
        op_type, target, precision, layout, alias
    )
