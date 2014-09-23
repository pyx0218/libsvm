#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict

if sys.version_info[0] >= 3:
	xrange = range

def exit_with_help(argv):
	print("""\
Usage: {0} dataset output 

This script transform the input dataset to the libsvm input format
""".format(argv[0]))
	exit(1)

def main(argv=sys.argv):
	argc = len(argv)
	if argc != 3:
		exit_with_help(argv)

	dataset = open(argv[1],'r')
	output = open(argv[2],'w')

	for line in dataset:
		tokens = line.split(',')
		s = str(ord(tokens[0][0]) - ord('A'))
		for i in xrange(1, len(tokens)):
			s += ' ' + str(i) + ':' + tokens[i]
		output.write(s)
	output.close()

if __name__ == '__main__':
	main(sys.argv)
