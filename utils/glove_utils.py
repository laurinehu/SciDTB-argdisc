import io


def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')
                '''
                input_ids = torch.tensor([tokenizer.encode(word, add_special_tokens=True)])
                last_hidden_states = model(input_ids)[0][0].detach().numpy()
                word_vec[word] = arrayToList(last_hidden_states)
                '''
                #print(word_vec[word])
                #print(len(word_vec[word]))
                #300 simple list

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    #print(len(word_vec))
    #print (word_vec)
    return word_vec


def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    #print(id2word, word2id)

    # list with words and dict  with 'word':i
    return id2word, word2id
