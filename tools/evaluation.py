#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict

if sys.version_info[0] >= 3:
	xrange = range

def exit_with_help(argv):
	print("""\
Usage: {0} test_file output_file 

This script evaluate the precision, recall and F-score of prediction.
""".format(argv[0]))
	exit(1)

def main(argv=sys.argv):
	argc = len(argv)
	if argc != 3:
		exit_with_help(argv)

	ground_truth = open(argv[1],'r')
	predict = open(argv[2],'r')
	
	truth = defaultdict(int)
	proposed = defaultdict(int)
	correct = defaultdict(int)
	
	for lt in ground_truth:
		line_t = lt.strip()
		true_label = line_t.split(' ')[0]
		truth[true_label] += 1
		lp = predict.readline()
		if lp == "":
			sys.stderr.write('''\
Error: the number of test data does not match predict output.''')
			sys.exit(-1)
		line_p = lp.strip()
		predict_label = line_p.split(' ')[0]
		proposed[predict_label] += 1
		if true_label == predict_label:
			correct[true_label] += 1

	ground_truth.close()
	predict.close()

	print('''\
label\tprecision\trecall\tF-score''')
	for label in truth:
		if proposed[label] > 0:
			precision = float(correct[label]) / proposed[label]
		else:
			precision = 0
		recall = float(correct[label]) / truth[label]
		if correct[label] > 0:
			f_score = 2*precision*recall / (precision + recall)
		else:
			f_score = 0
		print('''%s\t%8.3f%% (%i/%i)\t%8.3f%% (%i/%i)\t%8.3f%%''' 
				% (label, precision*100, correct[label], proposed[label],
					recall*100, correct[label], truth[label], f_score*100))

	total_data = sum(truth.values())
	total_correct = sum(correct.values())
	accuracy = float(total_correct) / total_data
	print('''Accuracy = %.3f%% (%i/%i)''' % (accuracy*100, total_correct, total_data))

if __name__ == '__main__':
	main(sys.argv)
