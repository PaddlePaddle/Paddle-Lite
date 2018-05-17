# Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io, re
import sys, os
import subprocess
import platform

COPYRIGHT = '''
/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
'''

LANG_COMMENT_MARK = None

NEW_LINE_MARK = None

COPYRIGHT_HEADER = None

if platform.system() == "Windows":
    NEW_LINE_MARK = "\r\n"
else:
    NEW_LINE_MARK = '\n'
    COPYRIGHT_HEADER = COPYRIGHT.split(NEW_LINE_MARK)[1]
    p = re.search('(\d{4})', COPYRIGHT_HEADER).group(0)
    process = subprocess.Popen(["date", "+%Y"], stdout=subprocess.PIPE)
    date, err = process.communicate()
    date = date.decode("utf-8").rstrip("\n")
    COPYRIGHT_HEADER = COPYRIGHT_HEADER.replace(p, date)


def generate_copyright(template, lang='C'):
    if lang == 'Python':
        LANG_COMMENT_MARK = '#'
    else:
        LANG_COMMENT_MARK = "//"

    lines = template.split(NEW_LINE_MARK)
    BLANK = " "
    ans = LANG_COMMENT_MARK + BLANK + COPYRIGHT_HEADER + NEW_LINE_MARK
    for lino, line in enumerate(lines):
        if lino == 0 or lino == 1 or lino == len(lines) - 1: continue
        if len(line) == 0:
            BLANK = ""
        else:
            BLANK = " "
        ans += LANG_COMMENT_MARK + BLANK + line + NEW_LINE_MARK

    return ans + "\n"


def lang_type(filename):
    if filename.endswith(".py"):
        return "Python"
    elif filename.endswith(".h"):
        return "C"
    elif filename.endswith(".c"):
        return "C"
    elif filename.endswith(".hpp"):
        return "C"
    elif filename.endswith(".cc"):
        return "C"
    elif filename.endswith(".cpp"):
        return "C"
    elif filename.endswith(".cu"):
        return "C"
    elif filename.endswith(".cuh"):
        return "C"
    elif filename.endswith(".go"):
        return "C"
    elif filename.endswith(".proto"):
        return "C"
    else:
        print("Unsupported filetype %s", filename)
        exit(0)


PYTHON_ENCODE = re.compile("^[ \t\v]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Checker for copyright declaration.')
    parser.add_argument('filenames', nargs='*', help='Filenames to check')
    args = parser.parse_args(argv)

    retv = 0
    for filename in args.filenames:
        fd = io.open(filename, encoding="utf-8")
        first_line = fd.readline()
        second_line = fd.readline()
        if "COPYRIGHT " in first_line.upper(): continue
        if first_line.startswith("#!") or PYTHON_ENCODE.match(
                second_line) != None or PYTHON_ENCODE.match(
                    first_line) != None:
            continue
        original_contents = io.open(filename, encoding="utf-8").read()
        new_contents = generate_copyright(
            COPYRIGHT, lang_type(filename)) + original_contents
        print('Auto Insert Copyright Header {}'.format(filename))
        retv = 1
        with io.open(filename, 'w') as output_file:
            output_file.write(new_contents)

    return retv


if __name__ == '__main__':
    exit(main())
