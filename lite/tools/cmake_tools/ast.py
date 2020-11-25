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

import logging

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
        assert self.cur == '"';
        self.cur_pos += 1;

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
        while self.cur.isalnum() or self.cur in ('_', ':',):
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

    def __repr__(self):
        str = "Kernel({op_type}, {target}, {precision}, {data_layout}, {alias}):".format(
            op_type = self.op_type,
            target = self.target,
            precision = self.precision,
            data_layout = self.data_layout,
            alias = self.alias,
        )

        str += '\n' + '\n'.join(repr(io) for io in self.inputs)
        str += '\n' + '\n'.join(repr(io) for io in self.outputs)
        str += '\n'
        return str


class RegisterLiteKernelParser(SyntaxParser):

    KEYWORD = 'REGISTER_LITE_KERNEL'

    def __init__(self, str):
        super(RegisterLiteKernelParser, self).__init__(str)

        self.kernels = []

    def parse(self, with_extra):
        find_registry_command = False
        extra_command = []
        # Get the code location of extra kernels registry
        # extra kernels registries are surrounded by
        # "#ifdef LITE_BUILD_EXTRA" and "#endif // LITE_BUILD_EXTRA"
        while self.cur_pos < len(self.str):
            start = self.str.find("#ifdef LITE_BUILD_EXTRA", self.cur_pos)
            if start != -1:
               self.cur_pos = start
               end = self.str.find("#endif  // LITE_BUILD_EXTRA", self.cur_pos)
               if end != -1:
                   extra_command += extra_command + list(range(start, end + 1))
                   self.cur_pos = end + len("#endif  // LITE_BUILD_EXTRA") -1
               else:
                   break
            else:
                break
        self.cur_pos = 0
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                #print 'str ', start, self.str[start-2: start]
                if start != 0 and '/' in self.str[start-2: start]:
                    '''
                    skip commented code
                    '''
                    self.cur_pos = start + 1
                    continue
                # if with_extra == "OFF", extra kernels will not be parsed
                if with_extra != "ON" and start in extra_command:
                    self.cur_pos = start + len(self.KEYWORD) -1
                    continue
                self.cur_pos = start
                k = KernelRegistry()
                self.kernels.append(self.parse_register(k))
            else:
                break

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


        # eat input and output
        while self.cur_pos < len(self.str):
            self.eat_point()
            self.eat_spaces()
            self.eat_word()
            assert self.token in ('BindInput', 'BindOutput', 'SetVersion', 'BindPaddleOpVersion', 'Finalize')
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
                self.eat_left_parentheses()
                self.eat_str()
                self.eat_comma()
                self.eat_spaces()
                self.eat_word()
                self.eat_right_parentheses()
                self.eat_spaces()
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

    def parse(self):
        while self.cur_pos < len(self.str):
            start = self.str.find(self.KEYWORD, self.cur_pos)
            if start != -1:
                #print 'str ', start, self.str[start-2: start]
                if start != 0 and '/' in self.str[start-2: start]:
                    '''
                    skip commented code
                    '''
                    self.cur_pos = start + 1
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


if __name__ == '__main__':
    with open('/home/chunwei/project2/Paddle-Lite/lite/kernels/arm/activation_compute.cc') as f:
        c = f.read()
        kernel_parser = RegisterLiteKernelParser(c)

        kernel_parser.parse()

#        for k in kernel_parser.kernels:
#            print k
