# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Name: ast.py
Usage: parser kernel registries from .cc source files into python struct `KernelRegistry`,
       we will generate `all_kernel_faked.cc` by calling this module. 
"""
import logging
import sys


class SyntaxParser(object):
    def __init__(self, str):
        self.str = str
        self.cur_pos = 0
        self.N = len(self.str)
        self.token = ''

    def eat_char(self):
        self.cur_pos += 1

    def eat_str(self):
        '''
        "xx"
        '''
        self.token = ''
        assert self.cur == '"'
        self.cur_pos += 1

        assert self.cur_pos < self.N
        while self.cur != '"':
            self.token += self.cur
            self.cur_pos += 1
            assert self.cur_pos < self.N
        assert self.cur == '"'
        self.cur_pos += 1
        #logging.warning('get: %s' % self.token)

    def eat_word(self):
        self.token = ''
        str = ''
        while self.cur.isalnum() or self.cur in (
                '_',
                ':', ):
            self.token += self.cur
            self.forward()

        #logging.warning('get: %s' % self.token)

    def eat_left_parentheses(self):
        '''
        (
        '''
        self.assert_is('(')
        self.token = '('
        self.forward()
        #logging.warning('get: %s' % self.token)

    def eat_right_parentheses(self):
        '''
        )
        '''
        self.assert_is(')')
        self.token = ')'
        self.forward()
        #logging.warning('get: %s' % self.token)

    def eat_left_brace(self):
        '''
        {
        '''
        self.assert_is('{')
        self.token = '{'
        self.forward()
        #logging.warning('get: %s' % self.token)

    def eat_right_brace(self):
        '''
        }
        '''
        self.assert_is('}')
        self.token = '}'
        self.forward()
        #logging.warning('get: %s' % self.token)

    def eat_comma(self):
        '''
        ,
        '''
        self.assert_is(',')
        self.token = ','
        self.forward()
        #logging.warning('get: %s' % self.token)

    def eat_spaces(self):
        '''
        eat space like string.
        '''
        while self.cur_pos < len(self.str):
            if self.cur in (' ', '\t', '\n'):
                self.forward()
            else:
                break

    def eat_point(self):
        '''
        .
        '''
        self.assert_is('.')
        self.token = '.'
        self.forward()
        #logging.warning('get: %s' % self.token)

    def eat_any_but_brace(self):
        '''
        anything but {}
        '''
        start = self.cur_pos
        while self.cur not in ('{', '}'):
            self.cur_pos += 1

        self.token = self.str[start:self.cur_pos]
        #logging.warning('get: %s' % self.token)

    def eat_semicolon(self):
        '''
        ;
        '''
        self.assert_is(';')
        self.token = ';'
        self.forward()
        #logging.warning('get: %s' % self.token)

    def assert_is(self, w):
        assert self.cur == w, "token should be %s, but get %s" % (w, self.cur)

    @property
    def cur(self):
        assert self.cur_pos < self.N
        return self.str[self.cur_pos]
        #logging.warning('get: %s' % self.token)

    def forward(self):
        self.cur_pos += 1


class IO:
    def __init__(self):
        self.name = ''
        self.type = ''

    def __repr__(self):
        return "- %s: %s" % (self.name, self.type)


class KernelRegistry:
    def __init__(self):
        self.op_type = ''
        self.target = ''
        self.precision = ''
        self.data_layout = ''
        self.class_ = ''
        self.alias = ''
        self.inputs = []
        self.outputs = []
        self.op_versions = []

    def __repr__(self):
        str = "Kernel({op_type}, {target}, {precision}, {data_layout}, {alias}):".format(
            op_type=self.op_type,
            target=self.target,
            precision=self.precision,
            data_layout=self.data_layout,
            alias=self.alias, )

        str += '\n' + '\n'.join(repr(io) for io in self.inputs)
        str += '\n' + '\n'.join(repr(io) for io in self.outputs)
        str += '\n'
        return str


class SubgraphBridgeRegistry:
    def __init__(self):
        self.op_type = ''
        self.target = ''


class RegisterLiteKernelParser(SyntaxParser):

    KEYWORD = 'REGISTER_LITE_KERNEL'

    def __init__(self, str):
        super(RegisterLiteKernelParser, self).__init__(str)

        self.kernels = []

    def parse(self, with_extra, enable_arm_fp16):
        find_registry_command = False
        extra_command = []
        arm_fp16_command = []
        # Get the code location of extra kernels registry
        # extra kernels registries are surrounded by
        # "#ifdef LITE_BUILD_EXTRA" and "#endif // LITE_BUILD_EXTRA"
        tmp_pos = self.cur_pos
        while tmp_pos < len(self.str):
            start = self.str.find("#ifdef LITE_BUILD_EXTRA", tmp_pos)
            if start != -1:
                tmp_pos = start
                end = self.str.find("#endif  // LITE_BUILD_EXTRA", tmp_pos)
                if end != -1:
                    extra_command += extra_command + list(
                        range(start, end + 1))
                    tmp_pos = end + len("#endif  // LITE_BUILD_EXTRA") - 1
                else:
                    break
            else:
                break
        # Get the code location of arm_fp16 kernels registry
        # arm_fp16 kernels registries are surrounded by
        # "#ifdef ENABLE_ARM_FP16" and "#endif"
        tmp_pos = self.cur_pos
        while tmp_pos < len(self.str):
            start = self.str.find("#ifdef ENABLE_ARM_FP16", tmp_pos)
            if start != -1:
                tmp_pos = start
                end = self.str.find("#endif  // ENABLE_ARM_FP16", tmp_pos)
                if end != -1:
                    arm_fp16_command += arm_fp16_command + list(
                        range(start, end + 1))
                    tmp_pos = end + len("#endif  // ENABLE_ARM_FP16") - 1
                else:
                    break
            else:
                break
        self.cur_pos = 0
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                #print 'str ', start, self.str[start-2: start]
                if start != 0 and '/' in self.str[start - 2:start]:
                    '''
                    skip commented code
                    '''
                    self.cur_pos = start + 1
                    continue
                # if with_extra == "OFF", extra kernels will not be parsed
                if with_extra.upper() != "ON" and start in extra_command:
                    self.cur_pos = start + len(self.KEYWORD) - 1
                    continue
                # if enable_arm_fp16 == "OFF", arm_fp16 kernels will not be parsed
                if enable_arm_fp16.upper(
                ) != "ON" and start in arm_fp16_command:
                    self.cur_pos = start + len(self.KEYWORD) - 1
                    continue
                self.cur_pos = start
                k = KernelRegistry()
                self.kernels.append(self.parse_register(k))
            else:
                break

    def pick_kernel_class(self, op_name, device_target, data_type, layout_type,
                          alias_name, first_flag, file_path):
        """pick the actual used kernel on the basis of kernel attribute information.

        self.str() stores the original source content. Kernel attributes include op_name,
        device_target, data_type, layout_type and alias_name and these five attributes is
        unique with regard to each kernel. We first divide the whole code into two sections,
        one is the kernel class definition code and the other one is kernel register code
        indentified by `REGISTER_LITE_KERNEL` and only class name alias indentified by
        `using` or `typedef` keyword is allowed between them. We subtract the kernel class
        definition code and class name alias code when first_flag is `True` and register
        code is obtained whenever first_flag is `True` or `False`.

        Args:
            op_name:  the 1st attribute of the kernel, such as `conv2d`.
            device_target:  the 2nd attribute of the kernel, such as `kARM`.
            data_type:  the 3rd attribute of the kernel, such as `kFloat`.
            layout_type:  the 4th attribute of the kernel, such as `kNCHW`.
            alias_name:  the 5th attribute of the kernel, such as `def`.
            first_flag:  the first time to pick the some kind of kernel.
            file_path:  the path to store the tailored kernel result.
        Returns:
            no val is returned as the `res_str` is stored into file_path.
        """
        f = open(file_path, 'a+')
        dst = f.read()
        res_str = ""
        main_idx = self.str.find("}  // namespace paddle", 0)
        if main_idx != -1:
            main_idx += len("}  // namespace paddle")
        else:
            main_idx = self.str.find("} /* namespace paddle */", 0)
            if main_idx != -1:
                main_idx += len("} /* namespace paddle */")
            else:
                sys.exit(-1)
        if first_flag == "True":
            res_str += self.str[:main_idx] + "\n"
            self.cur_pos = main_idx + 1
            while self.cur_pos < len(self.str):
                start = self.str.find("typedef", self.cur_pos)
                if start != -1:
                    end = self.str.find(";", start)
                    if end != -1:
                        res_str += self.str[start:end + len(";")] + "\n"
                        self.cur_pos = end + len(";")
                    else:
                        break
                else:
                    break
            self.cur_pos = main_idx + 1
            while self.cur_pos < len(self.str):
                start = self.str.find("using", self.cur_pos)
                if start != -1:
                    end = self.str.find(";", start)
                    if end != -1:
                        res_str += self.str[start:end + len(";")] + "\n"
                        self.cur_pos = end + len(";")
                    else:
                        break
                else:
                    break
        self.cur_pos = main_idx + 1
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                end = self.str.find(".Finalize();", self.cur_pos)
                if end != -1:
                    end += len(".Finalize();")
                else:
                    break
                left_brace = self.str.find("(", start)
                pos = left_brace + 1
                brace_num = 1
                while True:
                    if self.str[pos] == ')':
                        brace_num -= 1
                    elif self.str[pos] == '(':
                        brace_num += 1
                    if brace_num == 0:
                        break
                    pos += 1
                right_brace = pos
                kernel_attr = self.str[left_brace + 1:right_brace].replace(
                    '\n', '').replace(' ', '').split(",")
                if len(kernel_attr) != 6:
                    sys.exit(1)
                op_name_ = kernel_attr[0]
                device_target_ = kernel_attr[1]
                data_type_ = kernel_attr[2]
                layout_type_ = kernel_attr[3]
                alias_name_ = kernel_attr[5]
                if ((op_name_ == op_name) and
                    (device_target_ == device_target) and
                    (data_type_ == data_type) and
                    (layout_type_ == layout_type) and
                    (alias_name_ == alias_name)):
                    res_str += self.str[start:end] + "\n\n"
                self.cur_pos = end + 1
            else:
                break
        f.write(res_str)
        f.close()

    def eat_class(self):
        start = self.cur_pos
        self.eat_word()
        stack = ''
        if self.cur == '<':
            stack = stack + '<'
            self.forward()
            while stack:
                if self.cur == '<':
                    stack = stack + '<'
                elif self.cur == '>':
                    stack = stack[1:]
                else:
                    pass
                self.forward()
        self.token = self.str[start:self.cur_pos]

    def parse_register(self, k):

        self.eat_word()
        assert self.token == self.KEYWORD
        self.eat_spaces()

        self.eat_left_parentheses()
        self.eat_spaces()

        self.eat_word()
        k.op_type = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_word()
        k.target = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_word()
        k.precision = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_word()
        k.data_layout = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_class()
        k.class_ = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_word()
        k.alias = self.token
        self.eat_spaces()

        self.eat_right_parentheses()
        self.eat_spaces()

        def eat_io(is_input, io):
            self.eat_left_parentheses()
            self.eat_str()
            io.name = self.token
            self.eat_comma()
            self.eat_spaces()

            self.eat_left_brace()
            self.eat_any_but_brace()
            io.type = self.token
            self.eat_right_brace()
            self.eat_spaces()
            self.eat_right_parentheses()
            self.eat_spaces()

        def eat_op_version(io):
            self.eat_left_parentheses()
            self.eat_str()
            io.name = self.token
            self.eat_comma()
            self.eat_spaces()
            self.eat_word()
            io.version = self.token
            self.eat_right_parentheses()
            self.eat_spaces()

        # eat input and output
        while self.cur_pos < len(self.str):
            self.eat_point()
            self.eat_spaces()
            self.eat_word()
            assert self.token in ('BindInput', 'BindOutput', 'SetVersion',
                                  'BindPaddleOpVersion', 'Finalize')
            io = IO()

            if self.token == 'BindInput':
                eat_io(True, io)
                k.inputs.append(io)
            elif self.token == 'BindOutput':
                eat_io(False, io)
                k.outputs.append(io)
            elif self.token == 'SetVersion':
                self.eat_left_parentheses()
                self.eat_str()
                self.version = self.token
                self.eat_right_parentheses()
                self.eat_spaces()
            # skip `BindPaddleOpVersion` command during parsing kernel registry 
            elif self.token == 'BindPaddleOpVersion':
                # eg BindPaddleOpVersion("fill_constant", 1)
                eat_op_version(io)
                k.op_versions.append(io)
            else:
                self.eat_left_parentheses()
                self.eat_right_parentheses()
                self.eat_semicolon()
                self.eat_spaces()
                return k
                break


class RegisterLiteOpParser(SyntaxParser):

    KEYWORD = 'REGISTER_LITE_OP'

    def __init__(self, str):
        super(RegisterLiteOpParser, self).__init__(str)
        self.ops = []

    def parse(self, with_extra):
        extra_command = []
        while self.cur_pos < len(self.str):
            start = self.str.find("#ifdef LITE_BUILD_EXTRA", self.cur_pos)
            if start != -1:
                self.cur_pos = start
                end = self.str.find("#endif  // LITE_BUILD_EXTRA",
                                    self.cur_pos)
                if end != -1:
                    extra_command += extra_command + list(
                        range(start, end + 1))
                    self.cur_pos = end + len("#endif  // LITE_BUILD_EXTRA") - 1
                else:
                    break
            else:
                break
        self.cur_pos = 0
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                #print 'str ', start, self.str[start-2: start]
                if start != 0 and '/' in self.str[start - 2:start]:
                    '''
                    skip commented code
                    '''
                    self.cur_pos = start + 1
                    continue
                # if with_extra == "OFF", extra kernels will not be parsed
                if with_extra != "ON" and start in extra_command:
                    self.cur_pos = start + len(self.KEYWORD) - 1
                    continue
                self.cur_pos = start
                self.ops.append(self.__parse_register())
            else:
                break
        return self.ops

    def __parse_register(self):
        self.eat_word()
        assert self.token == self.KEYWORD
        self.eat_spaces()

        self.eat_left_parentheses()
        self.eat_spaces()

        self.eat_word()
        return self.token


class RegisterSubgraphBridgeParser(SyntaxParser):
    KEYWORD = 'REGISTER_SUBGRAPH_BRIDGE'

    def __init__(self, str):
        super(RegisterSubgraphBridgeParser, self).__init__(str)
        self.subgraph_bridge = []

    def parse(self):
        self.cur_pos = 0
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                #print 'str ', start, self.str[start-2: start]
                if start != 0 and '/' in self.str[start - 2:start]:
                    '''
                    skip commented code
                    '''
                    self.cur_pos = start + 1
                    continue
                self.cur_pos = start
                k = SubgraphBridgeRegistry()
                self.subgraph_bridge.append(self.parse_register(k))
            else:
                break

    def parse_register(self, k):
        self.eat_word()
        assert self.token == self.KEYWORD
        self.eat_spaces()

        self.eat_left_parentheses()
        self.eat_spaces()

        self.eat_word()
        k.op_type = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_word()
        k.target = self.token
        self.eat_comma()
        self.eat_spaces()

        return k


class RegisterNNadapterBridgeParser(SyntaxParser):
    KEYWORD = 'USE_SUBGRAPH_BRIDGE'

    def __init__(self, str):
        super(RegisterNNadapterBridgeParser, self).__init__(str)
        self.subgraph_bridge = []

    def parse(self):
        self.cur_pos = 0
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                #print 'str ', start, self.str[start-2: start]
                if start != 0 and '/' in self.str[start - 2:start]:
                    '''
                    skip commented code
                    '''
                    self.cur_pos = start + 1
                    continue
                self.cur_pos = start
                for k in self.parse_register():
                    self.subgraph_bridge.append(k)
            else:
                break

    def parse_register(self):

        ks = list()

        self.eat_word()
        assert self.token == self.KEYWORD
        self.eat_spaces()

        self.eat_left_parentheses()
        self.eat_spaces()

        self.eat_word()
        op_type = self.token
        self.eat_comma()
        self.eat_spaces()

        self.eat_word()
        self.eat_comma()
        self.eat_spaces()
        '''
        "xx, yy"
        '''
        self.token = ''
        assert self.cur == '"'
        self.cur_pos += 1

        assert self.cur_pos < self.N
        while self.cur != ')':
            if (self.cur == ','):
                temp = SubgraphBridgeRegistry()
                temp.op_type = op_type
                temp.target = self.token
                ks.append(temp)
                self.token = ''
                self.cur_pos += 1
            else:
                if (self.cur != '"' and self.cur != ' ' and self.cur != '\n'):
                    self.token += self.cur
                self.cur_pos += 1
            assert self.cur_pos < self.N
        assert self.cur == ')'
        temp = SubgraphBridgeRegistry()
        temp.op_type = op_type
        temp.target = self.token
        ks.append(temp)
        self.eat_right_parentheses()
        self.eat_spaces()
        self.eat_semicolon()
        self.eat_spaces()

        return ks


if __name__ == '__main__':
    with open('/Paddle-Lite/lite/kernels/arm/conv_compute.cc') as f:
        c = f.read()
        kernel_parser = RegisterLiteKernelParser(c)
        kernel_parser.pick_kernel_class(
            "conv2d", "kARM", "kFloat", "kNCHW", "def", "True",
            "/Paddle-Lite/build.lite.android.armv8.clang/conv_compute.cc")
