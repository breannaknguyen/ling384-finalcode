import math
from probabilistic_earley import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#######################################
# Code was adapted from LING 227 HW 5 #
#######################################

def pretty_print_chart(chart):
    for index, row in enumerate(chart):
        print("ROW " + str(index))
        for cell in row:
            print(cell)
        print("")

def file2grammar(filename):
    grammar_dict = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            input = line.split()
            try:
                key = tuple(input[2:])
                value = [input[1], math.log(float(input[0]))]

                if key in grammar_dict:
                    grammar_dict[key].append(value)
                else:
                    grammar_dict[key] = [value]
            except (ValueError, IndexError) as e:
                print(f"Skipping invalid line: {line}")
                continue

    return grammar_dict

def convert_rhs_to_lhs(grammar_rhs_map):
    # for compatibility with continuation matrix
    grammar_lhs_map = {}
    for rhs, entries in grammar_rhs_map.items():
        for lhs, logprob in entries:
            prob = math.exp(logprob)
            grammar_lhs_map.setdefault(lhs, []).append([prob, list(rhs)])
    return grammar_lhs_map

def init_chart(sentence_length):
    chart = []
    for i in range(sentence_length):
        row = []
        for j in range(sentence_length):
            row.append({})
        chart.append(row)
    
    return chart

def initialize_diagonal(chart, grammar, sentence):
    for i, word in enumerate(sentence):
        cell = {}

        for key, values in grammar.items():
            if key == tuple([word]):
                for production in values:
                    nonterminal = production[0]
                    logprob = production[1]
                    cell[nonterminal] = {
                        "logprob": logprob,
                        "backpointer": [word]
                    }

        # to support unary rules
        added = True
        while added:
            added = False
            current_keys = list(cell.keys())
            for B in current_keys:
                unary_rhs = (B,)
                if unary_rhs in grammar:
                    for A, logprob in grammar[unary_rhs]:
                        total_logprob = cell[B]['logprob'] + logprob
                        if A not in cell or total_logprob > cell[A]['logprob']:
                            cell[A] = {
                                'logprob': total_logprob,
                                'backpointer': [[i, i], B]
                            }
                            added = True

        chart[i][i] = cell

    return chart


def logsumexp(a, b):
    # 
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))
    

def fill_in_cell(chart, grammar, index1, index2):

    cell = {}
    for i in range(index1, index2):
        for j in range(index1, index2):
            if j >= i:
                left_nt_dict = chart[index1][j]
                right_nt_dict = chart[i+1][index2]

                for key1 in left_nt_dict:
                    for key2 in right_nt_dict:
                        combined_nt = tuple([key1, key2]) 

                        if combined_nt in grammar:
                            leftprob = left_nt_dict[key1]['logprob']
                            rightprob = right_nt_dict[key2]['logprob']
                            for production in grammar[combined_nt]:
                                new_nt = production[0]
                                logprob = production[1]
                                combined_logprob = leftprob + rightprob + logprob

                                if new_nt not in cell:
                                    cell[new_nt] = {
                                        'logprob': combined_logprob,
                                        'backpointer': [[[index1, j], key1], [[i+1, index2], key2]]
                                    }
                                else:
                                    cell[new_nt]['logprob'] = logsumexp(
                                        cell[new_nt]['logprob'],
                                        combined_logprob
                                    )

    added = True
    while added:
        added = False
        current_keys = list(cell.keys())
        for B in current_keys:
            unary_rhs = (B,)
            if unary_rhs in grammar:
                for A, logprob in grammar[unary_rhs]:
                    total_logprob = cell[B]['logprob'] + logprob
                    if A not in cell or total_logprob > cell[A]['logprob']:
                        cell[A] = {
                            'logprob': total_logprob,
                            'backpointer': [[index1, index2], B]
                        }
                        added = True 
    chart[index1][index2] = cell

    return chart
        
def fill_in_initialized_chart(chart, grammar):
    chart_length = len(chart)

    for i in range(1, chart_length):
        diag_length = chart_length - i

        for j in range(diag_length):
            start = j
            end = j + i
            chart = fill_in_cell(chart, grammar, start, end)

    return chart

def children_of_node(chart, index1, index2, nonterminal):
    entry = chart[index1][index2][nonterminal]
    backpointer = entry['backpointer']

    # terminal
    if index1 == index2 and isinstance(backpointer, list) and isinstance(backpointer[0], str):
        return [nonterminal, backpointer]

    # support unary rules
    if isinstance(backpointer[1], str) and isinstance(backpointer[0], list) and isinstance(backpointer[0][0], int):
        return [nonterminal, children_of_node(chart, backpointer[0][0], backpointer[0][1], backpointer[1])]

    # support binary rules
    left = backpointer[0]
    right = backpointer[1]
    left_child = children_of_node(chart, left[0][0], left[0][1], left[1])
    right_child = children_of_node(chart, right[0][0], right[0][1], right[1])
    return [nonterminal, left_child, right_child]

def tree_from_chart(chart):
    return children_of_node(chart, 0, len(chart[0]) - 1, 'S')

def prob_from_chart(chart):
    cell = chart[0][-1]
    prob = cell['S']['logprob']

    return prob

def parse_sentence(sentence, grammar_filename):
    words = sentence.split()

    grammar = file2grammar(grammar_filename)
    chart = init_chart(len(words))
    chart = initialize_diagonal(chart, grammar, words)
    chart = fill_in_initialized_chart(chart, grammar)

    tree = tree_from_chart(chart)
    prob = prob_from_chart(chart)

    return chart, tree, prob

def cky_incremental(sentence, grammar_filename):
    words = sentence
    n = len(words)

    # load grammar in RHS‑map format
    grammar = file2grammar(grammar_filename)

    # initialise empty chart and the diagonal for all words
    chart = init_chart(n)
    chart = initialize_diagonal(chart, grammar, words)

    # init prefox probability list
    prefix_probs = [0.0] * n

    # fill chart incrementally, right boundary r grows 0..n‑1
    for r in range(n):
        # fill every span (l, r] with l < r
        for l in range(r - 1, -1, -1):
            chart = fill_in_cell(chart, grammar, l, r)

        # after completing column r, record P(prefix) = exp(best logP of S 0..r)
        if 'S' in chart[0][r]:
            prefix_probs[r] = math.exp(chart[0][r]['S']['logprob'])
        else:
            prefix_probs[r] = 0.0 

    return chart, prefix_probs

def create_left_corner_matrix_CKY(grammar):
    # to be used in second incremental parser
    list_nonterminals = list(grammar.keys()) + ["_LEX_"]
    nonterminal2index = {nt: idx for idx, nt in enumerate(list_nonterminals)}

    n = len(list_nonterminals)
    P_L = [[0.0 for _ in range(n)] for _ in range(n)]

    lex_index = nonterminal2index["_LEX_"]

    for lhs, rulelist in grammar.items():
        lhs_idx = nonterminal2index[lhs]
        for prob, rhs in rulelist:
            left = rhs[0]
            if left in nonterminal2index:                   # non‑terminal left corner
                P_L[lhs_idx][nonterminal2index[left]] += prob
            else:                                           # TERMINAL left corner
                P_L[lhs_idx][lex_index] += prob             # all go to _LEX_

    return P_L, nonterminal2index

def cky_incremental2(sentence, grammar_filename):

    # load grammar in RHS→LHS format
    grammar_rhs = file2grammar(grammar_filename)

    # convert to LHS → [prob, RHS] format for continuation matrix
    grammar_lhs = convert_rhs_to_lhs(grammar_rhs)

    # compute left-corner continuation matrix
    P_L, nt2idx = create_left_corner_matrix_CKY(grammar_lhs)
    R_L = recursive_matrix(P_L)
    root_row = R_L[nt2idx['ROOT']]
    continuation_mass = {A: root_row[i] for A, i in nt2idx.items()}

    # initialize chart and diagonal using RHS-style grammar
    n = len(sentence)
    chart = init_chart(n)
    chart = initialize_diagonal(chart, grammar_rhs, sentence)

    # prefix probabilities container
    prefix_probs = [0.0] * n

    # incremental parsing
    for r in range(n):

        # fill chart for all spans ending at r
        for l in range(r - 1, -1, -1):
            chart = fill_in_cell(chart, grammar_rhs, l, r)

        for l in range(r + 1):
            for rr in range(l, r + 1):
                if chart[l][rr]:
                    for A, entry in chart[l][rr].items():
                        logprob = entry['logprob']

    # compute prefix probability
        prefix_prob = 0.0
        for A, entry in chart[0][r].items():
            inside = math.exp(entry['logprob'])
            cont = continuation_mass.get(A, 0.0)
            weighted = inside * cont
            prefix_prob += weighted

        prefix_probs[r] = prefix_prob

    return chart, prefix_probs

def surprisal_list(prefix_probs):
    # compute surprisal values using standard formula
    s     = []
    prev  = 1.0
    for p in prefix_probs:
        if p == 0.0:
            s.append(float('inf'))
        else:
            s.append(-math.log2(p) + math.log2(prev))
            prev = p
    return s

def plot_surprisal_cky(words, surprisals):
    # plot from surprisal list
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
    ax.set_ylabel("surprisal")
    ax.set_ylim(0, 30)
    plt.tight_layout()
    fig = ax.get_figure()

    return fig

