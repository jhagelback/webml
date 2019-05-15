from os import listdir
from os.path import isfile, isdir
from pathlib import Path


# Scans the main directory and all sub-directories for JS files.
def scan_files(path):
    print("Starting scan")

    scan = [path]
    files = []
    while len(scan) > 0:
        p = scan[0]
        del scan[0]

        ldir = listdir(p)
        for f in ldir:
            cp = p + "/" + f
            if isfile(cp) and f.endswith(".js"):
                files.append(cp)
            if isdir(cp):
                scan.append(cp)

    # Sort files
    files.sort()

    for f in files:
        print(f)

    print("Scan done")
    return files


# Combines all JS files into one
def combine_files(files, output_path, filename, version):
    f = output_path + "/" + filename + "-" + version + ".js"
    print("Combining js files into:", f)
    with open(f, "w") as outfile:

        # Write some header metainfo
        outfile.write("\"use strict\";\n\n")
        outfile.write("/* Library version */\n")
        outfile.write("var VERSION = \"" + version + "\";\n")
        # Combine all files into one
        for fname in files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print("Done")


# Run combiner
cwd = Path.cwd()
files = scan_files(str(cwd) + "/js")
combine_files(files, str(cwd), "webml", "0.38")
