import pickle
import tensorflow as tf
from Inducution_network import Contrast_induction
import random
import  numpy as np
from Tokenizer import  Data_preprocess

random.seed(1)

query_per_batch = 512 #n_query 
class_per_batch = 16  #n_class
max_query_pre_class = 32 # class 32
max_query_lenth = 20 # 
min_class_query = 3
valid_num_sample = 5
learning_rate=1e-5
# model = Contrast_induction(d_model=312, number_class=class_per_batch, max_query_lenth=20)

with open('./train_data/sent2id.p', 'rb') as file:
    sent2kid = pickle.load(file)

def split_Data(sent2id, min_class_query, valid_num_sample):
    id2sent = {}
    use_id2sent = {}
    use_sent2id = {}
    valid_data = []
    for k, v in sent2id.items():
        if v not in id2sent:
            id2sent[v] = [k]
        else:
            id2sent[v].append(k)
    for id, sent_list in id2sent.items():
        if len(sent_list) > min_class_query:
            temp_use_sent = sent_list
            if len(sent_list) >= valid_num_sample:
                random.shuffle(temp_use_sent)
                valid_data.append(temp_use_sent[0])
                temp_use_sent = temp_use_sent[1:]
            use_id2sent[id] = temp_use_sent
            for sent in temp_use_sent:
                use_sent2id[sent] = id
    return id2sent, use_sent2id, use_id2sent, valid_data
#
def creat_batch(n_class, n_query, max_query_pre_class, sent2kid, use_sent2kid, use_kid2sent):
    sample_class = set()
    temp_class = []
    temp_query = []
    temp_class_query = []
    while len(sample_class) < n_class:
        sample_query = random.sample(list(use_sent2kid.keys()), 1)[0]
        if sent2kid[sample_query] not in sample_class:
            temp_class.append(sent2kid[sample_query])
            temp_query.append(sample_query)
            sample_class.add(sent2kid[sample_query])
    while len(temp_query) < n_query:
        sample_query = random.sample(list(sent2kid.keys()), 1)[0]
        if sent2kid[sample_query] not in sample_class:
            temp_query.append(sample_query)
    for kid in temp_class:
        temp = [sent for sent in use_kid2sent[kid]]
        random.shuffle(temp)
        temp = temp[:max_query_pre_class]
        temp_class_query.append(temp)
    return temp_query, temp_class, temp_class_query




id2sent, use_sent2id, use_id2sent, valid_data = split_Data(sent2kid, min_class_query, valid_num_sample)

print("use_label", len(use_id2sent))
print("all_label", len(id2sent))
print("valid", len(valid_data))

with open('./train_data/id2sent_use_sent2id_use_id2sent_valid_data.p','wb') as file:
    pickle.dump((use_sent2id, use_id2sent, valid_data),file)




a=Contrast_induction(d_model=312, number_class=class_per_batch, max_query_lenth=20)
model=a.bulid_graph()
op=tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam',
)
model.summary()
model.compile(optimizer=op)

p=Data_preprocess()




def generate(n_class,n_query):
    i=0
    while(i<100):
        i+1
        temp_query,_,temp_class_query=creat_batch(n_class, n_query, max_query_pre_class, sent2kid, use_sent2id, use_id2sent)
        input_list=[]
        temp=[]
        
        for sent in  temp_query:
            temp.append(p.transform_data_one_piece(sentence=sent,lenth=20))
        temp=np.array(temp).reshape(-1,20)
        input_list.append(temp)
        input_list.append(np.zeros_like(temp))
        
        for i_class in temp_class_query:
            temp=[]
            for sent in i_class:
                temp.append(p.transform_data_one_piece(sentence=sent,lenth=20))
            temp=np.array(temp).reshape(-1,20)
        #     temp=np.array(temp)
            input_list.append(temp)
            input_list.append(np.zeros_like(temp))
        yield input_list

# print(creat_batch(class_per_batch, query_per_batch, max_query_pre_class, sent2kid, use_sent2id, use_id2sent))




model.fit_generator(generator=generate(class_per_batch,query_per_batch),shuffle=False,epochs=20)