# Based on code by Suhas Arehalli, which was 
# in turn based on algorithms from Stolcke (1995): https://aclanthology.org/J95-2002.pdf
# Note that this code does not implement all details of Stolcke's algorithm. The code is
# sufficient for the grammars used in our homework, but it will break on some other possible
# grammars (e.g., ones with empty categories or unit-production loops)

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#######################################
# Code was adapted from LING 384 HW 3 #
#######################################

# A Python class for a priority queue,
# which is like a list except that the values
# are kept sorted in order of priority
# Priority is established via a weight that is
# included whenever we add an item to the queue
# The items are listed in descending order of their weight
class PriorityQueue:

    # Initialize the queue
    def __init__(self):
        self.queue = []

    # Returns the highest-priority item from the queue, and removes
    # it from the queue
    def get(self):
        
        first = self.queue[0]
        self.queue = self.queue[1:]

        return first


    # Adds a new item to the queue
    def add(self, weight, item):

        index_to_insert = 0
        for index, (current_weight, current_item) in enumerate(self.queue):
            if current_weight < weight:
                index_to_insert = index

        self.queue = self.queue[:index_to_insert] + [(weight, item)] + self.queue[index_to_insert:]

    # Returns True if the queue is empty (i.e., has nothing in it);
    # returns False otherwise
    def empty(self):

        if len(self.queue) == 0:
            return True
        else:
            return False


# This function takes in a filename, such as pcfg_small.txt
# That file encodes a PCFG as follows:
# - Each line gives one rule of the PCFG
# - The line starts with a number, which is the probability
#   of that rule
# - Then, the first symbol after the number specifies the 
#   lefthand side of the rule
# - Then, the rest of the line specifies the righthand side 
#   of the rule
#
# For example, the first line in pcfg_small.txt stands for the 
# rule "S -> NP VP", which has a probability of 1.0
# 
# The dictionary that is returned should then have the
# the following structure:
# - Each lefthand side in the PCFG should be a key in 
#   the dictionary
# - The value for that key should be a list, where the
#   list has one element for each righthand side that
#   this lefthand side can have. Each element of that
#   list should be a list with 2 items: the probability
#   of the rule that has this righthand side, and the
#   righthand side itself (also expressed as a list)
#
# We previously asked you to write this function in HW2, so
# we won't ask you to implement it again! Instead, we've
# provided it for you here
def pcfg_file_to_dict(filename):
    fi = open(filename, "r")
    pcfg_dict = {}
    for line in fi:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
        parts = line.split()
        prob = float(parts[0])
        lhs = parts[1]
        rhs = parts[2:]

        if lhs not in pcfg_dict:
            pcfg_dict[lhs] = []
        pcfg_dict[lhs].append([prob, rhs])

    return pcfg_dict


# The next two functions (create_left_corner_matrix and
# recursive_matrix) are based on Section 4.5 of Stolcke (1995);
# they implement his solution for dealing with recursion in
# the grammar. For this homework, you don't need to worry
# about exactly what this code is doing, but if you want to understand
# it, you are welcome to ask us about it and/or to read Section 4.5
# of Stolcke (1995)!
def create_left_corner_matrix(grammar):

    list_nonterminals = list(grammar.keys())
    nonterminal2index = {}

    for index, nonterminal in enumerate(list_nonterminals):
        nonterminal2index[nonterminal] = index

    n_nonterminals = len(list_nonterminals)

    P_L = [[0 for i_inner in range(n_nonterminals)] for i_outer in range(n_nonterminals)]

    for lhs in grammar:
        lhs_index = nonterminal2index[lhs]
        for rule in grammar[lhs]:
            prob = rule[0]
            rhs = rule[1]
            left_corner = rhs[0]
            if left_corner in nonterminal2index:
                left_corner_index = nonterminal2index[left_corner]

                P_L[lhs_index][left_corner_index] += prob


    return P_L, nonterminal2index


# See comments about create_left_corner_matrix for 
# a description of this function
def recursive_matrix(P_L):

    np_matrix = np.array(P_L)

    R_L = np.linalg.inv(np.eye(len(P_L)) - np_matrix)

    return R_L.tolist()



# Takes in two cells from an Earley parse chart
# Returns True if they have the same lefthand side, 
# the same righthand side, the same dot position, and
# the same start index
# Else, returns False
# TODO 1.1: Replace "return False" with code
# that appropriately implements this function
def cells_same(cell1, cell2):
    if (cell1["LHS"] == cell2["LHS"] and
        cell1["RHS"] == cell2["RHS"] and
        cell1["dot"] == cell2["dot"] and
        cell1["start"] == cell2["start"]):
        return True
    else:
        return False




# Returns True if the cell indicates a completed
# constituent, False otherwise
# TODO 2.1: Replace "return False" with code 
# that implements this function. Hint: You may
# want to refer back to the slides to see when it is
# deemed that a cell triggers "complete"
def completable(cell):
    if cell["dot"] == len(cell["RHS"]):
        return True
    else:
        return False


# Takes in a cell that has been completed and a candidate cell
# Returns True if the candidate cell can have its dot advanced by
# the completion of completed_cell, False otherwise
# Note: Due to the structure of the code, we know that 
# candidate_cell_to_advance comes from the column of the chart that
# is necessary for it to be advanceable (i.e., it is the column corresponding
# to the "start" index for completed_cell). 
# So, all you need to check for is whether candidate_cell_to_advance is looking
# for the nonterminal that has been completed in completed_cell - but it's up
# to you to translate that into code!
# TODO 2.2: Replace "return False" with code 
# that implements this function.
def can_advance_via_completion(completed_cell, candidate_cell_to_advance):
    dot = candidate_cell_to_advance["dot"]
    rhs = candidate_cell_to_advance["RHS"]
    if dot < len(rhs) and rhs[dot] == completed_cell["LHS"]:
        return True
    else:
        return False
    




# Takes in two cells: A cell that has been completed, and a cell
# whose dot can be advanced due to the completion of the first cell
# Returns the new cell that can be created by advancing the
# dot of the second cell
# TODO 2.3: Fill in code in the middle of this function in order
# to have it return an appropriate cell
def new_cell_from_complete(completed_cell, cell_to_advance_dot_of):
    
    new_cell = {}

    # FILL IN CODE HERE THAT ADDS APPROPRIATE
    # VALUES TO new_cell
    new_cell["LHS"] = cell_to_advance_dot_of["LHS"]
    new_cell["RHS"] = cell_to_advance_dot_of["RHS"]
    new_cell["dot"] = cell_to_advance_dot_of["dot"] + 1
    new_cell["start"] = cell_to_advance_dot_of["start"]
    new_cell["gamma"] = cell_to_advance_dot_of["gamma"] * completed_cell["gamma"]
    new_cell["alpha"] = cell_to_advance_dot_of["alpha"] * completed_cell["gamma"]

    return new_cell




# Returns True if we are able to do a successful scan of cell
# A successful scan is when the item after the dot inside cell
# matches the true next word in the sentence, allowing us
# to add a new cell to the next column in the chart as a 
# a result of scanning
# TODO 3.1: Fill in this function by replacing "return False"
# with appropriate code. The arguments that are provided
# to the function should give you some hints - you will
# most likely need to use all four of these arguments!
def scannable(cell, pcfg, words, current_index_in_chart):
    dot = cell["dot"]
    rhs = cell["RHS"]

    if dot < len(rhs):
        next_symbol = rhs[dot]

        if next_symbol not in pcfg and current_index_in_chart < len(words):
            return next_symbol == words[current_index_in_chart]

    return False



# Takes in a cell that we have determined can be successfully scanned, 
# and returns the new cell that can be created by scanning it
# TODO 3.2: Fill in this function
def new_cell_from_scan(current_cell):

    new_cell = {}

    # FILL IN CODE HERE THAT ADDS APPROPRIATE
    # VALUES TO new_cell
    new_cell["LHS"] = current_cell["LHS"]
    new_cell["RHS"] = current_cell["RHS"]
    new_cell["dot"] = current_cell["dot"] + 1
    new_cell["start"] = current_cell["start"]
    new_cell["gamma"] = current_cell["gamma"]
    new_cell["alpha"] = current_cell["alpha"]

    return new_cell



# Given a sentence and a PCFG, creates the Earley parse chart for 
# that sentence using that PCFG
# Note that we are not storing any backpointers, because we are
# not trying to get trees for sentences; instead, we only
# need probabilities for words.
# Note also that we are using standard probabilities rather than log
# probabilities. In more practical settings, we would need log probabilities,
# but this homework is small-scale enough that standard probabilities 
# will work
def earley(sentence, pcfg):

    # Create the matrices of values that we will
    # use to deal with recursion
    P_L, nonterminal2index = create_left_corner_matrix(pcfg)
    R_L = recursive_matrix(P_L)

    # Split the sentence into words
    words = sentence.split()

    # Initialize the prefix probabilities
    prefix_probs = [0 for _ in range(len(words))]

    # Initialize the chart by creating all of 
    # the columns we will need
    # Each column is represented as an initially empty list of cells
    # The whole chart is then a list of these columns
    chart = [[] for _ in range(len(words) + 1)]

    # Create the initial cell
    # Each cell is a dictionary with 6 values:
    # - The lefthand side of the rule ("LHS"), which is
    #   a string denoting a nonterminal
    # - The righthand side of the rule ("RHS"), which is
    #   a list of strings (each string being a terminal
    #   or nonterminal)
    # - The position of the dot in the rule ("dot"), which is
    #   an integer; the dot is assumed to be before the element of
    #   the righthand side that is at this index (a dot at the start
    #   of a rule has index 0)
    # - The index of the column where this rule starts ("start"), which
    #   is an integer (the first column has index 0)
    # - The inner probability associated with this entry ("gamma"), which
    #   is a float
    # - The forward probability associated with this entry ("alpha"), which
    #   is a float
    initial_cell = {}
    initial_cell["LHS"] = "ROOT"
    initial_cell["RHS"] = ["S"]
    initial_cell["dot"] = 0
    initial_cell["start"] = 0
    initial_cell["gamma"] = 1.0
    initial_cell["alpha"] = 1.0

    # Add the initial cell to the chart
    chart[0].append(initial_cell)

    # Create the priority queue that we will use to deal with the
    # potentially-completable items
    complete_queue = PriorityQueue()


    # Fill in the columns one by one, from left to right
    # Note that the way we proceed through the chart is a bit different
    # from what we showed in class, in order to deal with some finicky
    # edge cases. Understanding the details of this procedure should not
    # be necessary for completing the homework.
    for current_index_in_chart in range(len(chart)):

        # First we take care of all potential completions
        while not complete_queue.empty():
            
            # Using the priority queue, we obtain the index in the chart
            # of the next chart cell that we should check as potentially
            # being completed
            weight, current_cell_index = complete_queue.get()
            current_cell = chart[current_cell_index[0]][current_cell_index[1]]
            
            # Check if current_cell indeed indicates a completed constituent
            if completable(current_cell):
                
                # We now need to find the potential "customers" for
                # the nonterminal we have just completed. These will
                # be the cells of the column in the parse chart where
                # current_cell started
                cells_to_check = chart[current_cell["start"]]

                # We iterate through the cells that are potential customers
                # and check for each one if it indeed is looking for the 
                # nonterminal that was just completed
                for cell_to_check in cells_to_check:

                    # Check whether current_cell is able to advance the dot of
                    # cell_to_check - if so, we can do a completion!
                    if can_advance_via_completion(current_cell, cell_to_check):

                        # We create the new cell that results from our completion
                        new_cell = new_cell_from_complete(current_cell, cell_to_check)

                        # We add this new cell to our chart, and also to our priority queue of
                        # possible completions (since the result of a completion can in turn
                        # trigger new completions of its own)
                        chart[current_index_in_chart].append(new_cell)
                        complete_queue.add(cell_to_check["start"], [current_index_in_chart, len(chart[current_index_in_chart])-1])


        # Next we take care of the predict steps
        # Recursion makes this part of the algorithm more complicated than what
        # we presented in class, so we have provided this part of the code in full

        # We only predict from the cells that are initially present; we don't iterate over
        # newly-added cells (that's because the R_L approach compiles together everything
        # that would otherwise need iteration)
        n_initially_there = len(chart[current_index_in_chart])
        for current_index_in_current_position in range(n_initially_there):

            # Retrieve the cell at this position
            current_cell = chart[current_index_in_chart][current_index_in_current_position]

            # Check if there is something after the dot (if not, we can't predict)
            if current_cell["dot"] < len(current_cell["RHS"]):

                # Check if the thing after the dot is a nonterminal (the way that's done here
                # is checking if it's a key in the dict "pcfg")
                # We can only predict if it is a nonterminal
                next_elt = current_cell["RHS"][current_cell["dot"]]
                if next_elt in pcfg:

                    # We now check for all other nonterminals if they can derive (recursively if
                    # necessary) from next_elt
                    for new_nonterminal in pcfg:

                        # The values in the matrix R_L are used to adjust the probabilities in
                        # ways that account for recursion
                        RL_value = R_L[nonterminal2index[next_elt]][nonterminal2index[new_nonterminal]]
                        if RL_value > 0:
                            to_add = pcfg[new_nonterminal]

                            # Add all of the new cells that result from prediction
                            for prob, cand_rhs in to_add:
                                new_cell = {}
                                new_cell["LHS"] = new_nonterminal
                                new_cell["RHS"] = cand_rhs
                                new_cell["dot"] = 0
                                new_cell["start"] = current_index_in_chart
                                new_cell["gamma"] = prob
                                new_cell["alpha"] = RL_value*current_cell["alpha"]*prob
                                
                                # Check if the potential new_cell is already present
                                new_cell_already_present = False
                                for index_to_check, existing_cell in enumerate(chart[current_index_in_chart]):
                                    if cells_same(new_cell, existing_cell):
                                        new_cell_already_present = True
                                        index_existing = index_to_check 

                                # If it's already present, we don't add a new copy. Instead, we just add
                                # the probability from the new version to the existing probability.
                                # But if it's not there already, we do add it
                                if new_cell_already_present:
                                    chart[current_index_in_chart][index_existing]["alpha"] = chart[current_index_in_chart][index_existing]["alpha"] + new_cell["alpha"]
                                else:
                                    chart[current_index_in_chart].append(new_cell)


        # Finally, we take care of the scans
        # We iterate over all cells in the current chart
        for current_cell in chart[current_index_in_chart]:

            # Check if a successful scan is possible for this cell
            # If so, we scan!
            if scannable(current_cell, pcfg, words, current_index_in_chart):

                # Create the new cell that results from scanning
                new_cell = new_cell_from_scan(current_cell)

                # Add the new cell to our chart
                # TODO 3.3: There is a small error in the line below! Fix it.
                chart[current_index_in_chart+1].append(new_cell)

                # Also add the new cell to our priority queue of possible
                # completions (since scanning can result in a completed constituent)
                complete_queue.add(new_cell["start"], [current_index_in_chart+1, len(chart[current_index_in_chart+1])-1])

                # Update the prefix probabilities
                prefix_probs[current_index_in_chart] += new_cell["alpha"]

    # Return our parse chart!
    return chart, prefix_probs

# Given an Earley chart, returns True if the chart
# indicates that the sentence is grammatical, False otherwise
# TODO 4.1: Complete this function by replacing "return False"
# with appropriate code
def grammatical(chart):
    last_column = chart[-1]
    for cell in last_column:
        if cell["LHS"] == "ROOT" and cell["dot"] == len(cell["RHS"]):
            return True
    return False


# Given a list of prefix probabilities, returns a list
# of the surprisals for all the words in the sentence
# TODO 4.2: Complete this function
def surprisals(prefix_probs):
    surprisal_values = []

    # FILL IN APPROPRIATE CODE TO
    # POPULATE THE LIST
    for i, prob in enumerate(prefix_probs):
        if prob == 0:
            surprisal_values.append(float('inf'))
        elif i == 0:
            surprisal_values.append(-math.log2(prob))
        else:
            surprisal_values.append(-math.log2(prob) + math.log2(prefix_probs[i - 1]))

    return surprisal_values

def plot_surprisal_earley(words, surprisals):
    # plot from surprisals
    if len(words) != len(surprisals):
        raise ValueError("words and surprisals must be the same length")

    surprisals = [0.0 if math.isinf(x) or math.isnan(x) else x for x in surprisals]
    df = pd.DataFrame({"pos": range(len(words)),
                       "word": words,
                       "surprisal": surprisals})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.barplot(x="pos", y="surprisal", data=df,
                     estimator=sum)
    ax.set_xticklabels(df["word"])
    ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=9, label_type="edge")
    ax.set_xlabel("")
    ax.set_ylim(0, 9)
    ax.set_ylabel("surprisal")
    plt.tight_layout()
    fig = ax.get_figure()

    return fig