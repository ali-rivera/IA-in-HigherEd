#!/usr/bin/env python

"""
remove-timestamps.py

takes in a file name and removes the YouTube timestamps, then saves the new file in a processed folder

can be edited to iterate through a folder and reformat all files
not doing this now because pdfs need to be figured out.
"""

import re
import sys

#get file name from user
filename = input("file name (do not include .txt) \n")

#get content from file
file = open(f"{filename}.txt", 'r')
content = file.read()
file.close()

#remove timestamps
new_content = re.sub(pattern='\n\d:\d\d|\n\d\d:\d\d', repl="", string=content)

#save new (clean) file in processed file
f=open(f"processed\\{filename}.txt", "a")
f.write(new_content)
f.close()