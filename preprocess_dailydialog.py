import json, pandas as pd, numpy as np, pickle
import torch
from transformers import BertModel, BertTokenizer

def preprocess_text(x):
    for punct in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
        x = x.replace(punct, ' ')
    
    x = ' '.join(x.split())
    x = x.lower()
    
    return x


def create_utterances(filename, split):
    sentences, act_labels, emotion_labels, speakers, conv_id, utt_id = [], [], [], [], [], []
    
    lengths = []
    with open(filename, 'r') as f:
        for c_id, line in enumerate(f):
            s = eval(line)
            for u_id, item in enumerate(s['dialogue']):
                sentences.append(item['text'])
                act_labels.append(item['act'])
                emotion_labels.append(item['emotion'])
                conv_id.append(split[:2] + '_c' + str(c_id))
                utt_id.append(split[:2] + '_c' + str(c_id) + '_u' + str(u_id))
                speakers.append(str(u_id%2))
                
                # u_id += 1
                
    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))
    data['act_label'] = act_labels
    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id
    
    return data


# def load_pretrained_glove():
#     print("Loading GloVe model, this can take some time...")
#     glv_vector = {}
#     f = open('F://glove.840B.300d.txt', encoding='utf-8')

#     for line in f:
#         values = line.split()
#         word = values[0]
#         try:
#             coefs = np.asarray(values[1:], dtype='float')
#             glv_vector[word] = coefs
#         except ValueError:
#             continue
#     f.close()
#     print("Completed loading pretrained GloVe model.")
#     return glv_vector

def encode_labels(encoder, l):
    return encoder[l]


def pad_data(texts:list):
    max_num_tokens = 250
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens, segments, input_masks = [], [], []
    for text in texts:
        indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens.append(indexed_tokens)
        segments.append([0]*len(indexed_tokens))
        input_masks.append([1]*len(indexed_tokens))
    
    tokens_tensor_list, segments_tensor_list, input_masks_tensor_list = [], [], []
    # padding & tensor 
    for j in range(len(tokens)):
        if max_num_tokens > len(tokens[j]):
            padding = [0] * (max_num_tokens - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        else:
            tokens[j] = tokens[j][:max_num_tokens]
            segments[j] = segments[j][:max_num_tokens]
            input_masks[j] = input_masks[j][:max_num_tokens]
    
        tokens_tensor_list.append(torch.tensor([tokens[j]]))
        segments_tensor_list.append(torch.tensor([segments[j]]))
        input_masks_tensor_list.append(torch.tensor([input_masks[j]]))
        
    
    return tokens_tensor_list, segments_tensor_list, input_masks_tensor_list


def into_sequence(input_tensor):
    print(type(input_tensor))
    tmp, ret = input_tensor.numpy().tolist(), []
    print(type(tmp))
    for i in range(tmp):
        ret.append(tmp[i])
        for word in tmp[i]:
            ret[i].append(word)
    print(np.array(ret).shape)
    return ret



if __name__ == '__main__':

    train_data = create_utterances('dailydialog/train_1.json', 'train')
    valid_data = create_utterances('dailydialog/dev_1.json', 'valid')
    test_data = create_utterances('dailydialog/test_1.json', 'test')
    
    ## encode the emotion and dialog act labels ##
    all_act_labels, all_emotion_labels = set(train_data['act_label']), set(train_data['emotion_label'])
    act_label_encoder, emotion_label_encoder, act_label_decoder, emotion_label_decoder = {}, {}, {}, {}

    for i, label in enumerate(all_act_labels):
        act_label_encoder[label] = i
        act_label_decoder[i] = label
    
    for i, label in enumerate(all_emotion_labels):
        emotion_label_encoder[label] = i
        emotion_label_decoder[i] = label

    # pickle.dump(act_label_encoder, open('dailydialog/act_label_encoder.pkl', 'wb'))
    # pickle.dump(act_label_decoder, open('dailydialog/act_label_decoder.pkl', 'wb'))
    # pickle.dump(emotion_label_encoder, open('dailydialog/emotion_label_encoder.pkl', 'wb'))
    # pickle.dump(emotion_label_decoder, open('dailydialog/emotion_label_decoder.pkl', 'wb'))

    train_data['encoded_act_label'] = train_data['act_label'].map(lambda x: encode_labels(act_label_encoder, x))
    test_data['encoded_act_label'] = test_data['act_label'].map(lambda x: encode_labels(act_label_encoder, x))
    valid_data['encoded_act_label'] = valid_data['act_label'].map(lambda x: encode_labels(act_label_encoder, x))

    train_data['encoded_emotion_label'] = train_data['emotion_label'].map(lambda x: encode_labels(emotion_label_encoder, x))
    test_data['encoded_emotion_label'] = test_data['emotion_label'].map(lambda x: encode_labels(emotion_label_encoder, x))
    valid_data['encoded_emotion_label'] = valid_data['emotion_label'].map(lambda x: encode_labels(emotion_label_encoder, x))
    
    
    ## tokenize all sentences ##
    # all_text = list(train_data['sentence'])
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(all_text)
    # pickle.dump(tokenizer, open('dailydialog/tokenizer.pkl', 'wb'))
    ## use bert tokenizer
    print("pad data")
    train_tokens, train_segments, train_inputmasks = pad_data(list(train_data['sentence']))
    valid_tokens, valid_segments, valid_inputmasks = pad_data(list(valid_data['sentence']))
    test_tokens, test_segments, test_inputmasks = pad_data(list(test_data['sentence']))

    print("Loading Bert")
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    # put everything on cuda
    if torch.cuda.is_available:
        for j in range(len(train_tokens)):
            train_tokens[j] = train_tokens[j].to('cuda')
            train_segments[j] = train_segments[j].to('cuda')
            train_inputmasks[j] = train_inputmasks[j].to('cuda')

        for j in range(len(valid_tokens)):
            valid_tokens[j] = valid_tokens[j].to('cuda')
            valid_segments[j] = valid_segments[j].to('cuda')
            valid_inputmasks[j] = valid_inputmasks[j].to('cuda')


        for j in range(len(test_tokens)):
            test_tokens[j] = test_tokens[j].to('cuda')
            test_segments[j] = test_segments[j].to('cuda')
            test_inputmasks[j] = test_inputmasks[j].to('cuda')
        model.to('cuda')
    
    print("preprocessing...")
    train_results, valid_results, test_results = [], [], []
    with torch.no_grad():
        print("Begin train set")
        for i in range(len(train_tokens)):
            train_outputs = model(train_tokens[i], token_type_ids=train_segments[i], attention_mask=train_inputmasks[i])
            train_results.append(train_outputs[0][:,0,:].cpu().numpy().tolist()[0]) #cls
            if i % 1000 == 0:
                print(i)
        print("Begin valid set")
        for i in range(len(valid_tokens)):
            valid_outputs = model(valid_tokens[i], token_type_ids=valid_segments[i], attention_mask=valid_inputmasks[i])
            valid_results.append(valid_outputs[0][:,0,:].cpu().numpy().tolist()[0])
            if i % 1000 == 0:
                print(i)
        print("Begin test set")
        for i in range(len(test_tokens)):
            test_outputs = model(test_tokens[i], token_type_ids=test_segments[i], attention_mask=test_inputmasks[i])
            test_results.append(test_outputs[0][:,0,:].cpu().numpy().tolist()[0])
            if i % 1000 == 0:
                print(i)

    train_sequence, valid_sequence, test_sequence = np.array(train_results), \
                                                    np.array(valid_results), \
                                                    np.array(test_results)

    print(type(train_sequence))
    print(type(train_sequence[0]))
    ## convert the sentences into sequences ##
    # train_sequence = tokenizer.texts_to_sequences(list(train_data['sentence']))
    # valid_sequence = tokenizer.texts_to_sequences(list(valid_data['sentence']))
    # test_sequence = tokenizer.texts_to_sequences(list(test_data['sentence']))
    
    # train_data['sentence_length'] = [len(item) for item in train_sequence]
    # valid_data['sentence_length'] = [len(item) for item in valid_sequence]
    # test_data['sentence_length'] = [len(item) for item in test_sequence]

    # train_sequence = pad_sequences(train_sequence, maxlen=max_num_tokens, padding='post')
    # valid_sequence = pad_sequences(valid_sequence, maxlen=max_num_tokens, padding='post')
    # test_sequence = pad_sequences(test_sequence, maxlen=max_num_tokens, padding='post')

    train_data['sequence'] = list(train_sequence)
    valid_data['sequence'] = list(valid_sequence)
    test_data['sequence'] = list(test_sequence)
    
    ## save the data in pickle format ##
    convSpeakers, convInputSequence, convActLabels, convEmotionLabels = {}, {}, {}, {}
    train_conv_ids, test_conv_ids, valid_conv_ids = set(train_data['conv_id']), set(test_data['conv_id']), set(valid_data['conv_id'])
    all_data = train_data.append(test_data, ignore_index=True).append(valid_data, ignore_index=True)
    
    print ('Preparing dataset. Hang on...')
    flag = False
    for item in list(train_conv_ids) + list(test_conv_ids) + list(valid_conv_ids):

        df = all_data[all_data['conv_id'] == item]
        
        convSpeakers[item] = list(df['speaker'])
        convInputSequence[item] = df['sequence']
        if not flag:
            print(type(df['sequence']))
            print(df['sequence'])
            print(convInputSequence[item])
            flag = True

        # convInputMaxSequenceLength[item] = max(list(df['sentence_length']))
        convActLabels[item] = list(df['encoded_act_label'])
        convEmotionLabels[item] = list(df['encoded_emotion_label'])
        
    pickle.dump([convSpeakers, convInputSequence, convActLabels, convEmotionLabels,
                 train_conv_ids, test_conv_ids, valid_conv_ids], open('dailydialog/daily_dialogue_bert.pkl', 'wb'))
    
    
    ## save pretrained embedding matrix ##
    # glv_vector = load_pretrained_glove()
    # word_vector_length = len(glv_vector['the'])
    # word_index = tokenizer.word_index
    # inv_word_index = {v: k for k, v in word_index.items()}
    # num_unique_words = len(word_index)
    # glv_embedding_matrix = np.zeros((num_unique_words+1, word_vector_length))

    # for j in range(1, num_unique_words+1):
    #     try:
    #         glv_embedding_matrix[j] = glv_vector[inv_word_index[j]]
    #     except KeyError:
    #         glv_embedding_matrix[j] = np.random.randn(word_vector_length)/200

    # np.ndarray.dump(glv_embedding_matrix, open('dailydialog/glv_embedding_matrix', 'wb'))
    # print ('Done. Completed preprocessing.')
