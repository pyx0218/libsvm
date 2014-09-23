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
	if argc < 3:
		exit_with_help(argv)

	dataset = open(argv[1],'r')
	output = open(argv[2],'w')
	
	labels = []
	attrs = defaultdict(list)
	for l in dataset:
		line = l.strip()
		tokens = line.split(' ')
		labels.append(tokens[0])
		for i in range(1, len(tokens)):
			attr_value = tokens[i].split(':')[1]
			attrs[i-1].append(float(attr_value))
	
	means = []
	stds = []
	for attr_values in attrs.itervalues():
		n = len(attr_values)
		mean = sum(attr_values)/n
		std = math.sqrt(sum((x-mean)**2 for x in attr_values) / n)
		means.append(mean)
		stds.append(std)
		print "%f %f" % (mean, std)

	for i in range(len(labels)):
		s = labels[i]
		for j in range(len(attrs)):
			attr = (attrs[j][i]-means[j]) / stds[j]
			s += ' ' + str(j+1) + ':' + str(attr)
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
				attr = (attr-means[i-1]) / stds[i-1]
				s += ' ' + str(i) + ':' + str(attr)
			s += '\n'
			test_output.write(s)
		test_output.close()

if __name__ == '__main__':
	main(sys.argv)
