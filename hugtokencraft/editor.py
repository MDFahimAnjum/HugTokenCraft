import os
import json

def get_top_tokens(tokenizer,k):
    # Get the vocabulary of the tokenizer
    vocab = tokenizer.get_vocab()

    # Filter out words containing "##"
    filtered_vocab = {word: count for word, count in vocab.items() if "##" not in word}

    # Sort the filtered vocabulary based on the frequency counts
    sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)

    # Get the top k most frequent words
    top_words = sorted_vocab[:k]

    # get the strings
    top_words_str={word for word,count in top_words}

    # Print the top k most frequent words
    #print("Top k most frequent words (excluding subwords):")
    #for word, count in top_words:
    #    print(f"{word}: {count}")

    return top_words_str

def expand_vocabulary(tokenizer,tokens_to_add):
    tokens_to_add=list(tokens_to_add)
    tokenizer.add_tokens(new_tokens=tokens_to_add,special_tokens=False)
    return tokenizer

def reduce_vocabulary(tokenizer,tokens_to_keep):
    # Get the vocabulary of the tokenizer
    vocab = tokenizer.get_vocab()

    # Get the special tokens from the tokenizer
    special_tokens = tokenizer.all_special_tokens

    # Delete tokens that are not special tokens or in the selected words list
    tokens_to_delete = [token for token in vocab.keys() if token not in special_tokens and token not in tokens_to_keep]
    for token in tokens_to_delete:
        del tokenizer.vocab[token]
    print(f"\nVocabulary reduction: Done\n Original vocabulary size: {len(vocab)}\n New vocabulary size: {len(tokenizer.get_vocab())}\n")
    return tokenizer

def save_tokenizer(tokenizer,tokenizer_path,model_max_length=None,isreduced=True):
    # Save the modified tokenizer
    tokenizer.save_pretrained(tokenizer_path)
    print("Tokenizer files created\n")
    
    if isreduced:
        print("Vocabulary was reduced. Requires json editing")
        # IMPORTANT: The special token ids are not fully updated yet. 
        #Need to edit the added_tokens.json  and tokenizer_config.json
        tokenizer_class = type(tokenizer)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        print("Tokenizer loaded\n Starting to edit the json files...")
        
        # Extract tokens and their corresponding token IDs
        vocab = tokenizer.vocab
        tokens_and_ids = [(token, vocab[token]) for token in vocab]
        special_tokens = tokenizer.all_special_tokens
        special_tokens_and_ids = {token: token_id for token, token_id in tokens_and_ids if token in special_tokens}
        print(" Target Special token mapping:")
        #print(special_tokens_and_ids)
        for token, token_id in special_tokens_and_ids.items():
            print(f"  Special Token: {token}, Token ID: {token_id}")
        token_path=tokenizer_path

        # Edit added_tokens.json =============
        print("\n 1. Editing added_tokens.json")
        f1_path = os.path.join(token_path,"added_tokens.json")
        if os.path.exists(f1_path):
            # Open the JSON file
            with open(f1_path, 'r') as file:
                fdata = json.load(file)
            fdata_keys=list(fdata.keys())
            # edit the file
            for sp_token in fdata_keys:
                fdata[sp_token]=special_tokens_and_ids.get(sp_token, None)
            # Save the updated data back to the file
            with open(f1_path, 'w') as file:
                json.dump(fdata, file, indent=2)
        else:
            print("added_tokens.json not found and skipped")

        # Edit tokenizer_config.json =============
        print(" 2. Editing tokenizer_config.json")    
        f1_path = os.path.join(token_path,"tokenizer_config.json")
        if os.path.exists(f1_path):
            # Open the JSON file
            with open(f1_path, 'r') as file:
                fdata = json.load(file)
            # edit
            all_token_decoders=fdata["added_tokens_decoder"]
            all_token_decoder_keys=list(all_token_decoders.keys())
            for i in range(len(all_token_decoder_keys)):
                token_decoder_key=all_token_decoder_keys[i]
                token_decoder=all_token_decoders[token_decoder_key]
                token_decoder_str=token_decoder["content"]
                token_decoder_key_new=special_tokens_and_ids[token_decoder_str]
                token_decoder_data=all_token_decoders[token_decoder_key]
                del all_token_decoders[token_decoder_key]
                all_token_decoders[str(token_decoder_key_new)]=token_decoder_data           
            fdata["added_tokens_decoder"]=all_token_decoders
            if not model_max_length is None: 
                fdata["model_max_length"]=model_max_length
            print(" 3. Model max length updated: "+str(fdata["model_max_length"]))
            # Save the updated data back to the file
            with open(f1_path, 'w') as file:
                json.dump(fdata, file, indent=2)
        else:
            print("tokenizer_config.json not found and skipped")

        print(" Editing done. \nTokenization model creation completed.")
    else:
        print("Vocabulary was expanded. Therefore, skipped json editing")
    return tokenizer_path

def load_tokenizer(tokenizer_class,tokenizer_path):
    #load
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
    #validate the special token ids
    validate_tokenizer(tokenizer)
    print(f"Tokenizer loaded with vocabulary size: {len(tokenizer.get_vocab())}")
    return tokenizer

def validate_tokenizer(tokenizer):
    vocab = tokenizer.vocab

    # Extract tokens and their corresponding token IDs
    tokens_and_ids = [(token, vocab[token]) for token in vocab]
    special_tokens = tokenizer.all_special_tokens
    special_tokens_and_ids = {token: token_id for token, token_id in tokens_and_ids if token in special_tokens}

    # Print special tokens and their corresponding token IDs
    #for token, token_id in special_tokens_and_ids:
    #    print(f"Special Token: {token}, Token ID: {token_id}")

    test1= tokenizer.mask_token_id == special_tokens_and_ids.get('[MASK]', None)
    test2= tokenizer.cls_token_id == special_tokens_and_ids.get('[CLS]', None)
    test3= tokenizer.pad_token_id == special_tokens_and_ids.get('[PAD]', None)
    test4= tokenizer.sep_token_id == special_tokens_and_ids.get('[SEP]', None)
    test5= tokenizer.unk_token_id == special_tokens_and_ids.get('[UNK]', None) 

    is_pass=test1 & test2 & test3 & test4 & test5

    if is_pass:
        print("\nTokenization model Validation: Passed\n")
    else:
        print("\nTokenization model Validation: Failed. STOP and retrain!\n")
    return is_pass

