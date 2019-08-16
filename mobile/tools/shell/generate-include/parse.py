import sys

output = ""
for line in sys.stdin:
    line.strip()
    tag = "\\"
    if tag in line:
        index = line.index("\\")
        line = line[:index]
    output += line
for line in output.split(" "):
    line = line.strip()
    if "/Applications" in line:
        continue
    if len(line) <= 0:
        continue
    if not line.endswith(".h"):
        continue
    if not line.startswith("../../../src/"):
        continue
    print(line[len("../../../src/"):])
