# Text-mining-and-Analytics-


# Text mining and analytic

Text mining includes techniques for mining and analyzing text data to discover interesting patterns, extract useful knowledge, and support decision making, with an emphasis on statistical approaches that can be generally applied to arbitrary text data in any natural language with no or minimum human effort.

This module will introduce the learner to text mining and text manipulation basics. We cover basics of text processing including regular expressions in the R and Python modules itself. Also, I talked about text classification in the machine learning module. Further, in this module, I will talk about further interesting topics in text mining such as n-gram models, Named Entity Recognition, Natural Language Processing, Sentiment Analysis, and Summarization.



![image](https://user-images.githubusercontent.com/67232573/114065421-83712200-984f-11eb-9b5e-7ed80c915049.png)





# N-gram Models

Next Word Prediction

Learning n-gram models

Text Generation using n-gram models

Handling low frequency n-grams

Google n-gram

Evaluation of n-gram models

Information Retrieval using language models

Query Likelihood Model

Smoothed Query Likelihood Model

Laplace Smoothing

Jelinek-Mercer Smoothing

Dirichlet Smoothing and Two-Stage Smoothing

Overall IR Language Model

Python code: Building N-Gram models

Python code: Next word prediction using 2-gram models (max prob)

Python code: Next word prediction using 2-gram models (Weighted random choice based on freq)

Python code: Creating Tri-grams and higher n-gram models

Python code: Generating text using n-gram models with n>=3

Python code: Laplace Smoothed n-grams

Python code: Computing perplexity




![image](https://user-images.githubusercontent.com/67232573/114065545-a3084a80-984f-11eb-9f0f-050d9c0bfc2e.png)




![image](https://user-images.githubusercontent.com/67232573/114065585-b1566680-984f-11eb-8973-9d175e0de7ef.png)



# Named Entity Recognition

What is NER?

Why is NER challenging?

Applications of NER

Annotation and Evaluation for NER

Broad Approaches for NER

Rule based Approaches for NER: List lookup approach

Rule based Approaches for NER: Shallow parsing approach

Rule based Approaches for NER: Shallow parsing approach with context

Learning based Approaches for NER

Python Code: Read text file, extract sentences and words

Python Code: Part of Speech Tagging and NER

Python Code: Chunking/NER visualization

Python Code: Get complete Person Names and Location Names from any text




![image](https://user-images.githubusercontent.com/67232573/114065907-ff6b6a00-984f-11eb-80bf-711108710317.png)





# Natural Language Processing

What is NLP?

List of NLP Tasks

Why is NLP challenging?

Tokenization

Lemmatization and Stemming

Sentence Segmentation

Phrase Identification

Word Sense Disambiguation: Part 1

Word Sense Disambiguation: Part 2

Parsing

Python Code: Word Tokenization with nltk

Python Code: Stemming and Lemmatization with nltk

Python Code: Tokenization, Word Counts, Stop Word removal, and Text Normalization using Italian recipes data

Python Code: Text Processing with Conference Abstracts Dataset

Python Code: Text Classification for Reuters Dataset using Scikit-Learn



![image](https://user-images.githubusercontent.com/67232573/114066017-1b6f0b80-9850-11eb-980d-947affaec9ac.png)





# Sentiment Analysis

Applications of Sentiment Analysis

Word Classification based Approach for Sentiment Analysis

Naïve Bayes for Sentiment Analysis

Challenges in Sentiment Analysis

Sentiment Lexicons

Learning Sentiment Lexicons: “and” and “but”

Learning Phrasal Sentiment Lexicons: Turney’s Algorithm

Learning Sentiment Lexicons: WordNet approach

Learning Sentiment Lexicons: Domain specific

Python Code: Basic Sentiment Analysis using Naive Bayes and sentiment dictionaries

Python Code: Sentiment Analysis on Movie Reviews Dataset

Python Code: Sentiment analysis on Twitter Data obtained via Tweepy



![image](https://user-images.githubusercontent.com/67232573/114066106-33468f80-9850-11eb-8e68-d03c6a92fc6f.png)






# Summarization

What is Summarization? What are its applications?

Genres and Types of Summaries

Position-based, cue phrase-based and word frequency-based approaches for extractive summarization

Lex Rank

Problems with Extractive Summarization Methods

Cohesion-based Methods

Lexical Chains Method for Extractive Summarization

Information Extraction based Method for Extractive Summarization

Interpretation Methods for Summarization

Multi-document Summarization

Evaluating Summaries – Extrinsic vs Intrinsic

Evaluating Summaries – ROUGE and BLEU

Python Code: Write a Simple Summarizer in Python from Scratch

Python Code: Text Summarization using Gensim (uses TextRank based summarization)

Python Code: Text Summarization using sumy (LSA, Word freq method, cue phrase method)

Python Code: LexRank using sumy

Python Code: Summarization using PyTeaser

Python Code: Text Rank using summa



![image](https://user-images.githubusercontent.com/67232573/114066221-56713f00-9850-11eb-957b-34bd50b3fcde.png)






# Topic Modeling

What are topic models? Why do you need them?

Plate diagrams, unigram models, mixture of unigrams

Application of topic modeling to matrices with high dimensionality

Singular Value Decomposition

Latent Semantic Indexing/Analysis (LSI/LSA) as an application of SVD

Latent Semantic Indexing/Analysis (LSI/LSA): Examples, Advantages and Drawbacks

Probabilistic Latent Semantic Analysis (PLSA)

Comparison between LSI and PLSA/PLSI

Motivation for LDA

Dirichlet Distributions

LDA Model Details

Comparison between various topic models: unigrams, mixture of unigrams, PLSI, LDA

LDA Hyper-parameters

Other Topic Models

Python Code: LDA using gensim

Python Code: LDA using scikit learn

Mini Project: Topic Modeling with Gensim – Loading data

Mini Project: Topic Modeling with Gensim – Pre-processing

Mini Project: Topic Modeling with Gensim – Building LDA Model

Mini Project: Topic Modeling with Gensim – Visualization

Mini Project: Topic Modeling with Gensim – Mallet and Hyper-parameter Tuning

Mini Project: Topic Modeling with Gensim – LDA Model analysis



![image](https://user-images.githubusercontent.com/67232573/114066288-6d179600-9850-11eb-9cc3-f930d23c70cb.png)





# Word Representation Learning

What are word representations? Where can you use word vectors?

Neural Network Language Model (NNLM)

Word2Vec

CBOW and Skip-gram

GloVe (Global vectors for word representation)

Python Code: Using gensim to train your first Word2Vec model

Python Code: Finding similar words using gensim Word2Vec model

Python Code: More stuff with word2vec models: Find odd one out, compute accuracy, get the actual vector, and save model.

Python Code: Another gensim model example using Text8 corpus

Python Code: GloVe Example

Python Code: Using Stanford’s GloVe Embedding




![image](https://user-images.githubusercontent.com/67232573/114066494-a4864280-9850-11eb-8236-7d60b359f1c1.png)




