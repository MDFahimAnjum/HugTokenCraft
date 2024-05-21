# HugTokenCraft
**HugTokenCraft** is a user-friendly Python library that simplifies the process of modifying the vocabulary of a PreTrainedTokenizer from HuggingFace Transformers, making it accessible without additional training. As of now, this was validated for BertTokenizer, which is word-piece-based vocabulary.

# Installation
### Install from PyPI
You can install HugTokenCraft using pip:

```bash
pip install hugtokencraft
```
### Install from source
```bash
git clone git@github.com/MDFahimAnjum/HugTokenCraft.git
cd HugTokenCraft
python setup.py install
```
# Usage

## 1. Reduce vocabulary
Let's take a pre-trained BertTokenizer which has 30,000 tokens and modify it to only keep 20 tokens
```python
#import library
from hugtokencraft import editor
from transformers import BertTokenizer
import os

#load BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#check
initial_vocab_size=len(tokenizer)
print(f"initial vocab size: {initial_vocab_size}")

#Target vocabulary
target_vocab_size=20
selected_words=editor.get_top_tokens(tokenizer,target_vocab_size)

#parameters
current_directory = os.getcwd()
# Define the path where you want to save the tokenizer
tokenizer_path = os.path.join(current_directory,"ModifiedTokenizer")
model_max_length=128

#reduce vocabulary
modified_tokenizer=editor.reduce_vocabulary(tokenizer,selected_words)
tokenizer_path=editor.save_tokenizer(modified_tokenizer,tokenizer_path,model_max_length)
modified_tokenizer=editor.load_tokenizer(type(tokenizer),tokenizer_path)

#check
new_vocab_size=len(modified_tokenizer)
print(f"new vocab size: {new_vocab_size} words")
```
## 2. Expand vocabulary
Let's take a pre-trained BertTokenizer and add two new tokens
```python

#import library
from hugtokencraft import editor
from transformers import BertTokenizer
import os

#load BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#check
initial_vocab_size=len(tokenizer)
print(f"initial vocab size: {initial_vocab_size}")

#Target vocabulary
selected_words_add={'hugtoken','hugtokencraft'}

#parameters
current_directory = os.getcwd()
# Define the path where you want to save the tokenizer
tokenizer_path = os.path.join(current_directory,"ModifiedTokenizer")


#expand vocabulary
modified_tokenizer=editor.expand_vocabulary(tokenizer,selected_words_add)
tokenizer_path=editor.save_tokenizer(modified_tokenizer,tokenizer_path,model_max_length=None,isreduced=False)
modified_tokenizer=editor.load_tokenizer(type(tokenizer),tokenizer_path)

#check
new_vocab_size=len(modified_tokenizer)
print(f"new vocab size: {new_vocab_size}")
```
# Notebook Example
You can also run the Python jupyter notebook examples directly by running example_notebook.ipynb

# Documentation

## `get_top_tokens()`
Obtains the _k_ most frequently used tokens from tokenizer vocabulary.

### Syntex
```python
token_set=get_top_tokens(tokenizer,k)
```
### Parameters
- tokenizer: _BertTokenizer_
    - Pre-trained Bert Tokenizer
- k: _int_
    - Desired number of tokens

### Returns
- token_list: _set_
    - Set of _k_ most frequent tokens

## `expand_vocabulary()`
Adds a set of new tokens to the vocabulary

### Syntex
```python
modified_tokenizer=expand_vocabulary(tokenizer,tokens_to_add)
```
### Parameters
- tokenizer: _BertTokenizer_
    - Pre-trained Bert Tokenizer
- tokens_to_add: _set_
    - Set of tokens to add

### Returns
- modified_tokenizer: _BertTokenizer_
    - Modified Bert Tokenizer

## `reduce_vocabulary()`
Removes all tokens execpt the given set of tokens from vocabulary

### Syntex
```python
modified_tokenizer=reduce_vocabulary(tokenizer,tokens_to_keep)
```
### Parameters
- tokenizer: _BertTokenizer_
    - Pre-trained Bert Tokenizer
- tokens_to_keep: _set_
    - Set of tokens to keep

### Returns
- modified_tokenizer: _BertTokenizer_
    - Modified Bert Tokenizer

## `save_tokenizer()`
Saves the modified tokenizer for use

### Syntex
```python
tokenizer_path=save_tokenizer(tokenizer,tokenizer_path,model_max_length=None,isreduced=True)
```
### Parameters
- tokenizer: _BertTokenizer_
    - Pre-trained Bert Tokenizer
- tokenizer_path: _str_
    - Location path to save the tokenizer
- model_max_length: _int_
    - New value of maximum token length
    - Defaults to None which means no change
- isreduced: _bool_
    - Whether the modified tokenizer was reduced
    - True if vocabulary was reduced (Default)
    - False if vocabulary was expanded   
### Returns
- tokenizer_path: _str_
    - Location path to save the tokenizer

## `load_tokenizer()`
Loads a tokenizer from a given path
### Syntex
```python
tokenizer=load_tokenizer(tokenizer_class,tokenizer_path)
```
### Parameters
- tokenizer_class: _type_
    - Class type of Tokenizer
- tokenizer_path: _str_
    - Location path to save the tokenizer
### Returns
- tokenizer: _tokenizer_class_
    - Tokenizer


## `validate_tokenizer()`
Simple sanity check for tokenizer 
### Syntex
```python
is_pass=validate_tokenizer(tokenizer)
```
### Parameters
- tokenizer: _BertTokenizer_
    - Pre-trained Bert Tokenizer
### Returns
- is_pass: _bool_
    - Valication result
    - True: validation passed
    - False: Validation failed

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Contributing

We welcome contributions! 