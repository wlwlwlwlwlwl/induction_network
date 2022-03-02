import codecs
import unicodedata


class Data_preprocess():
    def __init__(self, ):
        self.dict_path = './tiny_roberta/vocab.txt'
        self.token_dict = {}
        self.build_vocab()
        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'

    def build_vocab(self):
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

    @staticmethod
    def _is_space(ch):
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    def ourtokenize(self, text, add_cls=True, add_sep=True):
        R = []
        for c in text:
            if c in self.token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[PAD]')
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        if add_cls:
            R = ["[CLS]"] + R
        if add_sep:
            R = R + ["[SEP]"]
        return R

    def transform_data_one_piece(self, sentence, lenth=None):
        sent_id = [self.token_dict[vocab] for vocab in self.ourtokenize(sentence)]
        if lenth and len(sent_id) > lenth:
            sent_id = sent_id[:lenth]
        if lenth and len(sent_id) < lenth:
            sent_id = sent_id + (lenth - len(sent_id)) * [self.token_dict[self._token_pad]]
        return sent_id

    def transform_data(self, sentence_list, label_list, lenth):
        return_sentence = []
        return_label = []
        for i in range(len(sentence_list)):
            sent = ''.join(sentence_list[i])
            sent_id = [self.token_dict[vocab] for vocab in self.ourtokenize(sent)]
            if lenth and len(sent_id) > lenth:
                sent_id = sent_id[:lenth]
            if lenth and len(sent_id) < lenth:
                sent_id = sent_id + (lenth - len(sent_id)) * [self.token_dict[self._token_pad]]
            label_id = [0 for i in range(len(sent_id))]
            if label_list[i]:
                for idx in label_list[i]:
                    label_id[idx + 1] = 1
            return_sentence.append(sent_id)
            return_label.append(label_id)
        return return_sentence, return_label

    def tokens_to_id(self, tokens):
        return [self.token_dict.get(word, self.token_dict[self._token_unk]) for word in tokens]

    def encode(self, first_text, second_text, max_length=None, first_length=None, second_length=None):
        first_tokens = self.ourtokenize(first_text, add_cls=False, add_sep=False)
        if second_text is None:
            if max_length is None:
                first_tokens = first_tokens[:max_length - 2]
        else:
            second_tokens = self.ourtokenize(second_text, add_cls=False, add_sep=False)
            if max_length is not None:
                second_tokens = second_tokens[:max_length - 2]
        first_tokens = [self._token_cls] + first_tokens + [self._token_sep]
        first_tokens = self.tokens_to_id(first_tokens)
        if first_length is not None:
            first_tokens = first_tokens[:first_length]
            first_tokens.extend([self.token_dict[self._token_pad]] * (first_length - len(first_tokens)))
        first_segment_ids = [0] * len(first_tokens)

        if second_text is not None:
            second_tokens = second_tokens + [self._token_sep]
            second_tokens = self.tokens_to_id(second_tokens)
            if second_length is not None:
                second_tokens = second_tokens[:second_length]
                second_tokens.extend([self.token_dict[self._token_pad]] * (second_length - len(second_tokens)))
            second_segment_ids = [1] * len(second_tokens)
            first_tokens.extend(second_tokens)
            first_segment_ids.extend(second_segment_ids)
        return [first_tokens, first_segment_ids]