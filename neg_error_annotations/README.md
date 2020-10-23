Negation and machine translation
Annotation files - negation errors in evaluation data sets
June 1, 2020

These data files support the analyses described in 
"It's not a non-issue: Negation as an error source for machine translation"
accepted to EMNLP 2020.

Specifically, these are the files discussed in section 5 of the paper.

Source of data:
* Evaluation data sets from WMT 2018 and WMT 2019 Shared Tasks (news)
* Subset of sentences: only those with an English negation cue in the source sentence, as determined by an automatic negation cue detection system
* Five language pairs, always translating into English:
	German-English (de-en), WMT 2019
    Finnish-English (fi-en), WMT 2019
    Lithuanian-English (lt-en), WMT 2019
    Russian-English (ru-en), WMT 2019
    Turkish-English (tr-en), WMT 2018
* For each language pair, data (system output) from best performing system for that direction and year

Annotation process:
* Sort data by z-score (avg)
* For each sentence, compare reference translation and system output
* Annotate negation errors: binary annotation, 1 if there is a directly negation-related error in the system output, compared to the reference translation
* One annotator per language pair
* Some annotators made notes
* For German-English, we note the negation cue present in the source sentence (nicht, nein, kein, etc.)
* For Finnish, we note (1/0) whether a form of the negative auxiliary verb is present in the source sentence. In the same column, we note (with the word itself) when there is no negative auxiliary verb in the source sentence, but instead a negative adverbial, negative conjunction, or negative preposition.

File details:
* Naming convention includes: 
	shared task year for the data (e.g. wmt19)
	language pair and direction (e.g. deen for German-English)
    code corresponding to best performing system (e.g. Facebook_FAIR.6750)
* Field/column structure

a_line_number -- line number of sentence in original WMT test set
b_z_score -- average z-score for the sentence
c_source_sent -- source sentence
d_reference_sent -- reference translation
e_system_output -- system output translation
f_neg_error -- binary annotation re: presence of negation-related error
g_notes -- (optional) annotator notes
h_source -- (optional) information about negation cues in source sentence

Though the file structure is the same for all language pairs, not all language pairs include information in the two optional fields: notes and source cue information.


