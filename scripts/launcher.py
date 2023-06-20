from argparse import ArgumentParser
import os

parser = ArgumentParser()

parser.add_argument("--list_file", type=str, help="path to list file")
parser.add_argument("--line_id", type=int, help="list file line to launch")

args = parser.parse_args()

list_file = args.list_file
select_line = args.line_id

with open(list_file, "r") as fin: 
    lines = fin.readlines()

command = lines[select_line].strip()

os.system(command)
