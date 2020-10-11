Negation Cue Detection
===================================================================

## Requirements
Python 3.5+ (recommended: Python 3.6) \
Python packages: list of packages are provided in env-setup/requirements.txt file. \
Embedding: Download the file "glove.6B.300d.txt" from https://nlp.stanford.edu/projects/glove/ and put it in ./embeddings/globe directory

```bash
# Create virtual env (Assuming you have Python 3.5 or 3.6 installed in your machine) -> optional step
python3 -m venv your_location/negation-cue
source your_location/negation-cue/bin/activate

# Install required packages -> required step
pip install -r ./env-setup/requirements.txt
python -m spacy download en_core_web_sm
export HDF5_USE_FILE_LOCKING='FALSE'
```


## How to Run

- Example command line to train the cue-detector: 
```bash
  python train.py -c ./config/config.json 
```
  + Arguments:
	  - -c, --config_path: path to the configuration file, (required)
  
  *Note that: Training the system is optional, a trained model is already provided in "./model" directory. The pre-trained model can be used to predict negation cues in a text file (step is given below).
	
- Example command line to apply prediction on a given text file. 
```bash
  python predict.py -c ./config/config.json -i ./data/sample-io/input_file.txt -o ./data/sample-io/
```
  + Arguments:
	  - -c, --config-path: path to the configuration file; (required). Contains details parameter settings.
	  - -i, --input-file-path: path to the sample input file (text file, one sentence per line); (required)
	  - -o, --output-dir: path to the output directory (output files will be created in this directory); (required)
	  - --cd_sco_eval: if true, then creates a prediction file (in "./data/cd-sco-prediction/" by default) to evaluate cd-sco test corpus, (optional)
  
## Input and Output Files
- Sample input file:   \
./data/sample-io/input_file.txt (input file must contain one sentence per line)

- Sample output files: \
./data/sample-io/with_neg.txt (contains sentences with negation, one sentence per line) \
./data/sample-io/with_neg_cue_tags.txt (contains sentences (tokenized) with negation tag (Y if token is a negation cue, otherwise N) \
./data/sample-io/without_neg.txt (contains sentences without negation, one sentence per line)


## Citation

Negation cue detection is part of predicting scope of negation which is also part of the paper "Predicting the Focus of Negation: Model and Error Analysis". 
```bibtex
@inproceedings{hossain-etal-2020-predicting,
    title = "Predicting the Focus of Negation: Model and Error Analysis",
    author = "Hossain, Md Mosharaf  and
      Hamilton, Kathleen  and
      Palmer, Alexis  and
      Blanco, Eduardo",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.743",
    pages = "8389--8401",
    abstract = "The focus of a negation is the set of tokens intended to be negated, and a key component for revealing affirmative alternatives to negated utterances. In this paper, we experiment with neural networks to predict the focus of negation. Our main novelty is leveraging a scope detector to introduce the scope of negation as an additional input to the network. Experimental results show that doing so obtains the best results to date. Additionally, we perform a detailed error analysis providing insights into the main error categories, and analyze errors depending on whether the model takes into account scope and context information.",
}
```
