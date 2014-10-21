#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <unordered_map>
#include <math.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: svm-multiclass [options] training_set_file test_file output_file [model_file_prefix]\n"
	"options:\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-u multiclass_type : set type of multi-class classfication (default 0)\n"
    "   0 -- one versus all\n"
    "   1 -- cascading minority preference\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name, char *output_file_name, char *model_file_prefix);
void read_problem(const char *filename);
void do_cross_validation(const svm_problem *subprob,svm_parameter *subparam);
void predict(FILE *input, FILE *output);
int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;


struct svm_class
{
	double label;
	int count;
};

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_problem *subprobs;		// set by read_problem
struct svm_class *classes;
struct svm_model **submodels;
struct svm_node *x_space;
int nr_class;
int multiclass_type;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

struct svm_node *x;
int max_nr_attr = 64;

int predict_probability=0;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

int main(int argc, char **argv)
{
	char training_file_name[1024];
	char test_file_name[1024];
	char output_file_name[1024];
	char model_file_prefix[1024];
	const char *error_msg;

	parse_command_line(argc, argv, training_file_name, test_file_name, output_file_name, model_file_prefix);
	read_problem(training_file_name);
	error_msg = svm_check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	submodels = Malloc(svm_model *,nr_class);
	for(int i=0; i<nr_class; i++) {
		printf("train model %d...\n",i);
		if(cross_validation)
			do_cross_validation(&subprobs[i],&param);
		submodels[i] = svm_train(&subprobs[i],&param);
		char model_file_name[1024];
		sprintf(model_file_name,"%s%d",model_file_prefix,i);
		if(svm_save_model(model_file_name,submodels[i]))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}
		printf("save model %d\n",i);
	}

	FILE *input, *output;
	input = fopen(test_file_name,"r");
	if(input == NULL) {
		fprintf(stderr,"can't open input file %s\n",test_file_name);
		exit(1);
	}
	
	output = fopen(output_file_name,"w");
	if(output == NULL) {
		fprintf(stderr,"can't open output file %s\n",output_file_name);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));

	printf("predict...\n");
	predict(input, output);
	
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	for(int i=0;i<nr_class;i++) {
		svm_free_and_destroy_model(&submodels[i]);
	}
	free(submodels);
	free(x);
	fclose(input);
	fclose(output);

	return 0;
}

void do_cross_validation(const svm_problem *subprob,svm_parameter *subparam)
{
	double max_accuracy = 0;
	double max_precision = 0;
	double max_recall = 0;
	double max_f = 0;
	double c, g;

	svm_set_print_string_function(&print_null);

	printf("C\tgamma\tprecision\trecall\tf score\taccuracy\n");
	for(int i=0;i<3;i++) 
		for(int j=0;j<4;j++) {
			subparam->C = pow(10,i);
			subparam->gamma = pow(2,j);
			double *target = Malloc(double,subprob->l);
			svm_cross_validation(subprob,subparam,nr_fold,target);
			int total_correct = 0;
			int correct = 0;
			int predict = 0;
			int truth = 0;
			for(int k=0;k<subprob->l;k++) {
				if(target[k] == subprob->y[k]) {
					++total_correct;
					if(target[k] == 1)
						++correct;
				}
				if(target[k] == 1)
					++predict;
				if(subprob->y[k] == 1)
					++truth;
			}
			double accuracy = (double)total_correct/subprob->l;
			double precision = (double)correct/predict;
			double recall = (double)correct/truth;
			double f = 2*precision*recall/(precision+recall);
			printf("%f\t%f\t%.3f\t%.3f\t%.3f\t%.3f\n", subparam->C, subparam->gamma, precision, recall, f, accuracy);
			if(accuracy > max_accuracy) {
				c = subparam->C;
				g = subparam->gamma;
				max_accuracy = accuracy;
				max_precision = precision;
				max_recall = recall;
				max_f = f;
			}
			free(target);
		}
	subparam->C = c;
	subparam->gamma = g;
	printf("Cross Validation: C = %f, gamma = %f, Precision = %g%%, Recall = %g%%, F-Score = %g%%, Accuracy = %g%%\n", c, g, 100*max_precision, 100*max_recall, 100*max_f, 100*max_accuracy);
	void (*print_func)(const char*) = NULL;
	svm_set_print_string_function(print_func);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name, char *output_file_name, char *model_file_prefix)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;
	multiclass_type = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'u':
				multiclass_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				info = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc-2)
		exit_with_help();

	strcpy(input_file_name, argv[i++]);
	strcpy(test_file_name, argv[i++]);
	strcpy(output_file_name, argv[i++]);

	if(i<argc)
		strcpy(model_file_prefix,argv[i]);
	else
	{
		char *p = strrchr(argv[i-3],'/');
		if(p==NULL)
			p = argv[i-3];
		else
			++p;
		sprintf(model_file_prefix,"%s.model",p);
	}
}

int compare(const void *x, const void *y)
{
    svm_class elem1 = *(svm_class *) x;
    svm_class elem2 = *(svm_class *) y;
    return (elem1.count - elem2.count);
}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	std::unordered_map<double, int> labels;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(labels.count(prob.y[i]) > 0) {
			labels[prob.y[i]]++;
		}
		else {
			labels.insert(std::make_pair<double,int>(prob.y[i], 1));
		}

		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}
	
	classes = Malloc(struct svm_class, labels.size());
	i=0;
	for(std::unordered_map<double,int>::iterator it=labels.begin(); it!=labels.end(); it++) {
		classes[i].label = it->first;
		classes[i].count = it->second;
		i++;
	}

	nr_class = i;
	qsort(classes, nr_class, sizeof(svm_class), compare);

	printf("label\tcount\n");

	subprobs = Malloc(svm_problem, nr_class);
	for(int i=0; i<nr_class; i++) {
		printf("%g\t%d\n",classes[i].label,classes[i].count);
		subprobs[i].l = prob.l;
		subprobs[i].y = Malloc(double,prob.l);
		subprobs[i].x = Malloc(struct svm_node *,prob.l);
		for(int j=0; j<prob.l; j++) {
			subprobs[i].x[j] = prob.x[j];
			if(prob.y[j] == classes[i].label) {
				subprobs[i].y[j] = 1;
			}
			else subprobs[i].y[j] = -1;
		}
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	double *prob_estimates=(double *) malloc(nr_class*sizeof(double));

	int j;

	if(multiclass_type == 0) {
		fprintf(output,"One-versus-all\tCMP\tTrue label\n");
	}
	if(predict_probability)
	{
		for(j=0;j<nr_class;j++)
			fprintf(output,"\t%g",classes[j].label);
		fprintf(output,"\n");
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;
		
		int max_idx = 0;
		int cmp_idx;
		bool flag = false;
		for(j=0; j<nr_class; j++) {
			double *sub_prob_estimates=(double *) malloc(2*sizeof(double));
			double sub_predict_label = svm_predict_probability(submodels[j],x,sub_prob_estimates);
			prob_estimates[j] = sub_prob_estimates[0];
			if(multiclass_type == 1 && prob_estimates[j] > 0.5) {
				max_idx = j;
				break;
			}
			if(prob_estimates[j] > 0.5 && !flag) {
				cmp_idx = j;
				flag = true;
			}
			if(prob_estimates[j] > prob_estimates[max_idx]) {
				max_idx = j;
			}
			free(sub_prob_estimates);
		}
		if(!flag) cmp_idx = max_idx;

		predict_label = classes[max_idx].label;
		
		fprintf(output,"%g",predict_label);
	
		if (multiclass_type == 0) {
			fprintf(output,"\t%g\t%g",classes[cmp_idx].label,target_label);
		}

		if (multiclass_type == 0 && predict_probability)
		{
			for(j=0;j<nr_class;j++)
				fprintf(output,"\t%g",prob_estimates[j]);
		}
		fprintf(output,"\n");
		
		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	
	info("Accuracy = %g%% (%d/%d) (classification)\n",
		(double)correct/total*100,correct,total);
	
	free(prob_estimates);
}
