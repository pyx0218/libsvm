#!/usr/bin/env python

import os, sys
from subprocess import call

def exit_with_help(argv):
	print("""\
Usage: {0} training_file test_file multiclass_type
multiclass_type: 0: one versus one
1: one versus all
""").format(argv[0])
	exit(1)

def main(argv):
	if len(argv) != 4:
		exit_with_help(argv)
	
	if argv[3] == "0":
		u = 0
		model_file_prefix = argv[1] + ".model.one"
		output_file_prefix = argv[2] + ".output.one"
	elif argv[3] == "1":
		u = 1
		model_file_prefix = argv[1] + ".model.all"
		output_file_prefix = argv[2] + ".output.all"
	else:
		exit_with_help(argv)

	call(["rm", model_file_prefix+".*"])
	call(["rm", output_file_prefix+".*"])
	for c in range(-3, 4):
		for g in range(-1, 5):
			print ("C = %f, g = %f" % (10**c, 2**g))
			call(["/Users/panyanxi/Courses/Skewed Multiclass/libsvm/svm-train", "-q", "-u", str(u), "-c", str(10**c), "-g", str(2**g), \
					argv[1], model_file_prefix+"."+str(c)+"."+str(g)])
			call(["/Users/panyanxi/Courses/Skewed Multiclass/libsvm/svm-predict", argv[2], model_file_prefix+"."+str(c)+"."+str(g), \
					output_file_prefix+"."+str(c)+"."+str(g)])

if __name__ == '__main__':
	main(sys.argv)
