# Down the Garden Path with Probabilistic CKY and Earley Parsers
## Breanna K. Nguyen --- LING 384 Final Project Code Repository

### PCFGs
Custom PCFGs were created to reflect the structural ambiguities of three garden path sentence types:
- **MV/RR** (e.g., *"The horse raced past the barn fell."*)
- **NP/Z** (e.g., *"Because the nurse examined the patient recovered."*)
- **NP/S** (e.g., *"The journalist believed the senator lied."*)
Each PCFG allows both the ambiguous and disambiguated parses, along with simpler completions. See [`pcfg/`](./pcfg/) for grammar files.

### Penn Treebank-Derived Probabilities
Rule probabilities were estimated using relative frequencies from the **Penn Treebank**. A preprocessing script ([`penn_analysis.py`](./penn_analysis.py)) extracts and normalizes rule expansions for each nonterminal to create realistic, data-driven PCFGs.

### Earley Parser
The probabilistic Earley parser (in [`probabilistic_earley.py`](./probabilistic_early.py)) implements the prefix probability computation. The code was adapted from from LING 384 Homework 3.

### Incremental CKY Parsers

Two versions of CKY are implemented in [`cky.py`](./cky.py):
1. **Basic incremental CKY** – computes prefix probabilities from spans ending at each word.
2. **Continuation-mass CKY** – incorporates a left-corner continuation matrix for probabilistically grounded prefix estimates.

The code was adapted from LING 227 Homework 5.

### Main Notebook
[`main.ipynb`](./main.ipynb) contains the full pipeline for:
- Loading grammars
- Running Earley and CKY parsers on garden path vs disambiguated sentences
- Computing word-by-word prefix probabilities and **surprisal**
- Plotting the results

### Dependencies

To use the set of packages required to run the notebook:
```bash
pip install -r requirements.txt
