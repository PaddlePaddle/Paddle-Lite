import os
import sys
import subprocess

username = ""
email = ""
home = ""
desktop = "{}/Desktop".format(home)
dir_1 = "{}/1".format(desktop)
dir_2 = "{}/2".format(desktop)
src_dir = dir_1
dest_dir = dir_2
src_mobile_dir = "{}/paddle-mobile".format(src_dir)
dest_mobile_dir = "{}/paddle-mobile".format(dest_dir)

def clone_repo(dir):
    os.chdir(dir)
    os.system("git clone git@github.com:{}/paddle-mobile.git".format(username))
    os.chdir("{}/paddle-mobile".format(dir))
    os.system("git remote add upstream git@github.com:PaddlePaddle/paddle-mobile.git")
    os.system("git config user.name {}".format(username))
    os.system("git config user.email {}".format(email))

def get_output(command):
    out = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    return stdout.decode("utf-8").split("\n")

if __name__ == "__main__":
    # if not os.path.isdir(src_dir):
    #     print("dir 1 not found")
    #     sys.exit(-1)
    
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    if not os.path.isdir(dest_mobile_dir):
        clone_repo(dest_dir)
    sys.exit()
    
    items = []
    # items = ["metal/.gitignore", "metal/VideoSuperResolution"]
    os.chdir(src_mobile_dir)
    for line in get_output("git status --porcelain"):
        line = line.strip()
        items.append(line.split(" ")[-1])
    
    for item in items:
        src = item
        if len(src) <= 0:
            continue
        dest = dest_mobile_dir + "/" + item
        cmd = "cp -R " + src + " " + dest
        print(cmd)
        os.system(cmd)
