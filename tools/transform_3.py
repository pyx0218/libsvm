#!/usr/bin/env python

import sys, math, random
from os import listdir
from os.path import isfile, join

if sys.version_info[0] >= 3:
	xrange = range

def exit_with_help(argv):
	print("""\
Usage: {0} dataset num_attributes output 

This script transform the input dataset to the libsvm input format
""".format(argv[0]))
	exit(1)

def main(argv=sys.argv):
	argc = len(argv)
	if argc != 3:
		exit_with_help(argv)

	#dataset = open(argv[1],'r')
	output = open(argv[2],'w')

	features = []
	features.append({"vhigh":0, "high":1, "med":2, "low":3})
	features.append({"vhigh":0, "high":1, "med":2, "low":3})
	features.append({"2":0, "3":1, "4":2, "5more":3})
	features.append({"2":0, "4":1, "more":2})
	features.append({"small":0, "med":1, "big":2})
	features.append({"low":0, "med":1, "high":2})
	labels = {}
	label_num = 0
	#files = [ f for f in listdir(argv[1]) if isfile(join(argv[1], f))]
	#for f in files:
	for l in open(argv[1]):
		line = l.strip()
		tokens = line.split(',')
		label = tokens[len(tokens)-1]
		if label not in labels:
			labels[label] = label_num
			label_num += 1
		
		s = str(labels[label])

		for i in range(len(tokens)-1):
			s += ' ' + str(i+1) + ':' + str(features[i][tokens[i]])
		output.write(s+'\n')
	output.close()

if __name__ == '__main__':
	main(sys.argv)
