#!/usr/bin/env python3

from sys import exit, argv
import time

try: # attempt to import dependencies
	import pandas as pd
	import numpy as np
except ImportError: # if not found
	print("\nModules could not be loaded.")
	print("Ensure both `pandas` and `numpy` are installed before execution.\n")
	exit(1)

if len(argv) != 4: # if inappropriate arguments
	print("\nExecute the script as one of the below:\n")
	print(" $ ./id3_tree.py <File> <Holdout> <Print>")
	print(" $ ./id3_tree.py <Train> <Test> <Print>\n")
	print("In the former, a single data set is used for both train/test:")
	print(" <File> -- the .CSV file of examples")
	print(" <Holdout> -- the proportion of training examples (0.00..1.00)")
	print(" <Print> -- whether to print tree (1 = T, 0 = F)\n")
	print("For the latter, separate data sets are used instead:")
	print(" <Train> -- the .CSV file of training examples")
	print(" <Test> -- the .CSV file of testing examples")
	print(" <Print> -- whether to print tree (1 = T, 0 = F)\n")
	exit(1)
	
def id3(df, t, f):
	"""
	ID3 Decision Tree generator.
	
	Parameter:
	df -- the dataframe of training data
	t -- target attribute
	f -- list of attributes
	
	Return:
	root -- the fully formed decision tree
	"""
	root, ig = {}, {} # root node, IG dict
	attr = df.columns.drop(t) # get attribute set of df
	for a in attr:
		ig[a] = find_information_gain(df, t, a) # find IG of attr
	highest_ig = max(ig, key=lambda key: ig[key]) # return key of highest val
	s = make_split(df, highest_ig) # find splits on highest IG attr
	root = {highest_ig:{}} # found root for further branches	
	for v in s.keys(): # for each outcome of root
		df_branch = df.where(df[highest_ig] == v).dropna() # remove root node
		# if entropy of potential branch is zero, all outcomes same = term leaf
		if find_entropy(df_branch[t]) == 0:
			# add leaf branch
			root[highest_ig][v] = s[v][t].value_counts().idxmax()
		else: # otherwise branch has further subbranches = decision
			if len(attr) - 1 == 0: # if no more attr to divide on
				# entropy not 0, next branch isn't pure
				root[highest_ig][v] = s[v][t].value_counts().idxmax()
				return root
			else: # if more attr to split on, can recurse
				# recurse on split, dropping root attr
				root[highest_ig][v] = id3(s[v].drop(highest_ig, axis=1), t, f)
	return root
	
def find_entropy(t):
	"""
	Finds entropy of target attribute in training set.
	H(S) = \sum_{x\inX}{ -p(x)*log_2{p(x)} }
	
	Parameter:
	t -- target attribute
	
	Return:
	h -- entropy of target attribute
	"""
	h = 0
	v, n = np.unique(t, return_counts = True) # get values and distinct v
	for x in range(len(v)):
		px = n[x]/np.sum(n)
		h += -px * np.log2(px)
	return h
	
def find_information_gain(df, t, s):
	"""
	Finds information gain of target attribute in training set.
	IG(S,A) = H(S) - \sum_{t\inT}{ p(t) * H(t)} = H(S) - H(S|A)
	
	Parameter:
	df -- the dataframe of training data
	t -- target attribute
	s -- splitting attribute
	"""
	total_h = find_entropy(df[t]) # find entropy of entire system
	split_h = 0 # entropy after potential split
	v, n = np.unique(df[s], return_counts = True) # get values and distinct v
	for x in range(len(v)):
		pt = n[x]/np.sum(n)
		split = df.where(df[s] == v[x]).dropna()[t] # remove missing attrs
		split_h += pt * find_entropy(split)
	return total_h - split_h

def make_split(df, t):
	"""
	Splits a dataframe on attribute.
	
	Parameter:
	df -- the dataframe to split
	t -- target attribute to split upon
	
	Return:
	new_df -- split dataframe
	"""
	new_df = {}
	for df_key in df.groupby(t).groups.keys():
		new_df[df_key] = df.groupby(t).get_group(df_key)
	return new_df

def count_leaves(dt, c=[0,0]):
	"""
	Count number of non-leaf and leaf branches.
	
	Parameter:
	dt -- the decision tree
	c -- a counter
	
	Return:
	c -- a count for both non-leeaves and leaves
	"""
	c[0] += 1
	leaves = dt.keys()
	for leaf in leaves:
		branches = dt[leaf].values()
		for branch in branches:
			if isinstance(branch, dict):
				count_leaves(branch, c)
			else:
				c[1] += 1
	return c
	
def load_csv(f):
	"""
	Loads CSV file into pandas dataframe.
	.CSV file is organized such that decision is the last column and features
	are other columns. The first row is the name of decision and features.
	
	An example .CSV might be:
	F1 F2 F3 F4 F5 F6 D
	 0  0  1  1  0  1 1
	 1  0  1  1  0  1 0
	 1  0  0  0  1  1 0
	 1  1  0  1  1  0 1
	
	Where F1..Fn are attributes and D is the decision
	
	Parameter:
	f -- the filename for the .CSV file
	
	Return:
	df -- a dataframe of examples
	"""
	try:
		df = pd.read_csv(f, dtype=str) # open file as parse CSV into dataframe
	except:
		print("\nData could not be loaded, ensure the arguments are correct.\n")
		exit(1)
	print(f"{f} was successfully loaded.")
	return df

def holdout(df, p):
	"""
	Splits a dataframe of examples into training and testing data.
	
	Parameter:
	df -- a dataframe of examples
	p -- proportion of training vs testing (0.00..1.00]
	
	Return:
	train -- training examples
	test -- testing examples
	"""
	if 0.00 < p < 1.00:
		d = df.copy()
		train = d.sample(frac=p) # split, and randomize
		test = d.drop(train.index) # remove train data from df
		if len(test) == 0:
			print("Proportion of training examples is too high.\n")
			exit(1)
		return train, test
	else:
		print("\nThe proportion of training examples must be (0.00..1.00).\n")
		exit(1)

def find_accuracy(dt, t):
	"""
	Determines accuracy of the system.
	Accuracy = (1 - error) = (TP+TN)/(TP+TN+FP+FN)
	
	Parameter:
	dt -- the decision tree
	t -- a set of testing examples
	
	Return:
	accuracy -- how accurate the system is
	"""
	correct, total = 0, 0
	for _, e in t.iterrows():
		total += 1 # TP+TN+FP+FN
		if e[len(e)-1] == predict_decision(dt, e):
			correct += 1 # TP+TN
	return round(((correct/total)*100), 1)

def predict_decision(dt, e):
	"""
	Predicts decision on a testing example.
	
	Parameter:
	dt -- the decision tree
	e -- a testing example
	
	Return:
	decision -- a classification/decision
	"""
	split = list(dt.keys())[0]
	try:
		branch = dt[split][e[split]]
	except KeyError:
		return None
	if not isinstance(branch, dict):
		return branch
	return predict_decision(branch, e)

def print_tree(dt, indent=0):
	"""
	Prints decision tree in a better fashion.
	
	Parameter:
	dt -- the tree to display
	indent -- used internally for indentation
	"""
	for key, value in dt.items():
		print("  " * indent + str(key))
		if isinstance(value, dict): # if subdict
			print_tree(value, indent+1)
		else: # otherwise value
			print("  " * (indent+1) + str(value))

def print_statistics(dt, t, tr, te, trs, tes):
	"""
	Prints diagnostics regarding decision tree.
	
	Parameter:
	dt -- the decision tree
	t -- the time it took to generate dt
	tr -- classification ability of training data
	te -- classification ability of novel (test) data
	trs -- number of training examples
	tes -- number of testing examples
	"""
	s, d = count_leaves(dt) # splits and decisions
	print(f"Using {trs} training examples and {tes} testing examples.")
	print(f"Tree contains {s} non-leaf nodes and {d} leaf nodes.")
	print("Took {:.2f} seconds to generate.".format(t))
	print(f"Was able to classify {tr}% of training data.")
	print(f"Was able to classify {te}% of testing data.\n")

def get_data():
	"""
	Load CSV data depending on holdout or not.
	
	Return:
	train -- a set of training examples
	test -- a set of testing examples
	"""
	try: # singular set of examples
		h = float(argv[2])
		train, test = holdout(load_csv(argv[1]), h)
		print(f"\nUsing holdout style training, {h*100}% training data.")
	except ValueError: # separate train/test examples
		train, test = load_csv(argv[1]), load_csv(argv[2])
		print("\nUsing separate training and testing sets.")
	return train, test

if __name__ == '__main__':
	print()
	train, test = get_data()
	decision_name = train.columns[len(train.columns)-1]
	start_time = time.time()
	dt = id3(train, decision_name, train.columns[:-1]) # get decision tree
	end_time = time.time();
	if int(argv[3]) == 1:
		print(dt, "\n") # print decision tree as dict
		print_tree(dt) # print decision tree
		print()
	t = end_time-start_time
	tr_size = len(train)
	te_size = len(test)
	tr_ability = find_accuracy(dt, train)
	te_ability = find_accuracy(dt, test)
	print_statistics(dt, t, tr_ability, te_ability, tr_size, te_size)
	exit(0)