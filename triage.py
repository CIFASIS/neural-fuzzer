#!/usr/bin/python2

"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
"""

import os
import os.path
import argparse
import subprocess
import random

remove_nils = lambda xs: filter(lambda x: x <> '', xs)

"""
if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("seeds", help="", type=str, default=None)
    parser.add_argument("cmd", help="", type=str, default=None)
    parser.add_argument("-d", help="", type=int, default=5)
    parser.add_argument("-p", help="", action="store_true", default=False)


    options = parser.parse_args()
    seeds = options.seeds
    depth = options.d
    prune = options.p
    cmd = options.cmd
"""


def aflcount(cmd, seeds):
    cmd = "afl-count -m none -i "+seeds+" -o .afl-traces -- "+cmd
    #print(cmd)
    out = subprocess.check_output(cmd, shell=True)

    try:
      return int(out)
    except:
      return -1

def test(cmd, seeds):
    #print("\n")
    if seeds is None:
        os.system(cmd)
        return

    all_files = []

    for x, y, files in os.walk(seeds):
        nfiles = len(files)
        for f in files:
            f = f.replace("(","\(")
            f = f.replace(")","\)")
            f = f.replace("$","\$")
            f = f.replace(",","\,")

            all_files.append(x + "/".join(y) + "/" + f)


    random.shuffle(all_files)
    nfiles = len(all_files)

    for progress, testcase in enumerate(all_files):
        prepared_cmd = cmd.split("@@")
        prepared_cmd = prepared_cmd[0].split(
            " ") + [testcase] + prepared_cmd[1].split(" ")
        prepared_cmd = remove_nils(prepared_cmd)
        os.system(" ".join(prepared_cmd))
 
def triage(cmd, seeds, depth=5, prune=False):
    gdb_cmd = "env -i ASAN_OPTIONS='abort_on_error=1' gdb -batch -ex 'tty /dev/null' -ex run -ex bt 20 --args @@ 2> /dev/null"
    all_files = []
    dedup_files = dict()

    for x, y, files in os.walk(seeds):
        nfiles = len(files)
        for f in files:
            f = f.replace("(","\(")
            f = f.replace(")","\)")
            f = f.replace("$","\$")
            f = f.replace(",","\,")

            all_files.append(x + "/".join(y) + "/" + f)


    random.shuffle(all_files)
    #all_files = all_files[:1000]
    nfiles = len(all_files)

    for progress, testcase in enumerate(all_files):
        prepared_cmd = cmd.split("@@")
        prepared_cmd = prepared_cmd[0].split(
            " ") + [testcase] + prepared_cmd[1].split(" ")
        prepared_cmd = remove_nils(prepared_cmd)
        #print prepared_cmd
        out = subprocess.check_output(gdb_cmd.replace(
            "@@", " ".join(prepared_cmd)), shell=True)
        #print out
        backtrace = out.split("#")[1:]
        key = ""
        size = os.path.getsize(testcase)
        dkey = 0
        for x in backtrace:

            if dkey == depth:
                break

            if "??" in x or "__" in x:
                continue
            if " in " in x:
                x = remove_nils(x.split(" "))
                key = key + " " + x[3]
                dkey = dkey + 1

            else:
                x = remove_nils(x.split(" "))
                key = key + " " + x[1]
                dkey = dkey + 1

        # print key
        y = dedup_files.get(key, [])
        dedup_files[key] = y + [(testcase,size)]

    out = dict()

    for (k, xs) in dedup_files.items():
        #print "*"+k,
        xs = sorted(xs, key=lambda x: x[1])
        for x in xs[:1]:
            out[k] = x

        #if prune:
        #    for x in xs[1:]:
        #        os.remove(x[0])

        #print ""
    #print out
    return out
