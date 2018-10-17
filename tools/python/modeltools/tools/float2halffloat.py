# encoding:utf-8
import math
import re


def Real2HalfFloat(data):
    MINNUM = -65536
    MAXNUM = 65535
    FloatVal = 0
    if data:
        if data < MINNUM:
            data = MINNUM
        if data > MAXNUM:
            data = MAXNUM

        sign = 0
        if data < 0:
            sign = 1
            data = -data

        exp = math.floor((math.log2(data)))
        expout = exp + 16

        Mantial = round(data / pow(2, exp - 10)) - 1024

        if expout <= 0:
            FloatVal = 0
        else:
            FloatVal = sign * 32768 + expout * 1024 + Mantial
    return FloatVal


def ReadCfloatData(sourcefile):
    input = []
    with open(sourcfile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = re.sub('\s+', ' ', line)  # 两个数字间多个空格
            input.append(line.split(' '))
    destfile = sourcefile.replace('.dat', '')
    destfile = destfile.replace('.txt', '')
    destfile += 'Out.dat'
    with open(destfile, 'w') as fw:
        for i in range(len(input)):
            if len(input[i]) == 2:
                real = Real2HalfFloat(float(input[i][0]))
                imag = Real2HalfFloat(float(input[i][1]))
                result = real * 65536 + imag
                if imag and not real:
                    fw.write('0x0000' + "%X" % result + '\n')
                elif not imag and not real:
                    fw.write('0x00000000' + '\n')
                else:
                    fw.write('0x' + "%X" % result + '\n')
            elif len(input[i]) == 1:
                result = Real2HalfFloat(float(input[i][0]))
                if result:
                    fw.write('0x' + "%X" % result + '\n')
                else:
                    fw.write('0x0000' + '\n')


if __name__ == '__main__':
    print('Tips: Input number 0 if you want to exit!\n')
    while True:
        sourcfile = input("input source file:\n")
        if sourcfile is '0':
            break
        ReadCfloatData(sourcfile)
        print('Transfer Success!')
