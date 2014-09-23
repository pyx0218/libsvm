#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict

if sys.version_info[0] >= 3:
	xrange = range

def exit_with_help(argv):
	print("""\
Usage: {0} training_set training_output [test_set] [test_output]

This script transform the input dataset to the libsvm input format
""".format(argv[0]))
	exit(1)

def main(argv=sys.argv):
	argc = len(argv)
	if argc < 3:
		exit_with_help(argv)

	dataset = open(argv[1],'r')
	output = open(argv[2],'w')
	
	labels = []
	attrs = None
	for l in dataset:
		line = l.strip()
		tokens = line.split(' ')
		labels.append(tokens[0])
		if attrs is None:
			attrs = [[] for i in range(len(tokens)-1)]
		for i in range(1, len(tokens)):
			attr_value = tokens[i].split(':')[1]
			attrs[i-1].append(float(attr_value))
	
	min_values = []
	max_values = []
	for i in range(len(attrs)):
		max_values.append(max(attrs[i]))
		min_values.append(min(attrs[i]))
		if (max_values[i] - min_values[i]) > 0:
			attrs[i] = [(x - min_values[i]) / (max_values[i] - min_values[i]) for x in attrs[i]]
		else:
			attrs[i] = [0 for x in attrs[i]]
		print max_values[i], min_values[i]

	for i in range(len(labels)):
		s = labels[i]
		for j in range(len(attrs)):
			s += ' ' + str(j+1) + ':' + str(attrs[j][i])
		s += '\n'
		output.write(s)
	output.close()

	if argc > 3:
		if argc < 5:
			exit_with_help(argv)
		test_dataset = open(argv[3],'r')
		test_output = open(argv[4],'w')

		for l in test_dataset:
			line = l.strip()
			tokens = line.split(' ')
			s = tokens[0]
			for i in range(1, len(tokens)):
				attr = float(tokens[i].split(':')[1])
				attr = (attr - min_values[i-1]) / (max_values[i-1] - min_values[i-1])
				s += ' ' + str(i) + ':' + str(attr)
			s += '\n'
			test_output.write(s)
		test_output.close()

if __name__ == '__main__':
	main(sys.argv)
