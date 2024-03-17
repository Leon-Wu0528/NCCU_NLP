# NCCU_NLP
This repository contains my assignments and code for the Natural Language Processing course at NCCU.

## Folder Description
* Automated Fact Retrieval and Verification System:    
The course provides a fact database and a claim. Students are required to develop an automated fact retrieval and verification system to validate the truthfulness of 	the claim. If the claim can be "supported" or "opposed" by facts, the system must also provide evidence sentences by retrieving articles from the database.   
* homework:   
Classroom assignments include Chinese_word_segmentation, Naive_Bayes_Classifier, Stance_Detection.	

## Project Description
[brief report](Automated%20Fact%20Retrieval%20and%20Verification%20System/brief_report.pdf)


## Demo
```python
from transformers import BertTokenizer, TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 載入資料預處理結果
input_ids_files = ['input_ids_part1.npy', 'input_ids_part2.npy', 'input_ids_part3.npy', 'input_ids_part4.npy', 'input_ids_part5.npy', 'input_ids_part6.npy']
attention_masks_files = ['attention_masks_part1.npy', 'attention_masks_part2.npy', 'attention_masks_part3.npy', 'attention_masks_part4.npy', 'attention_masks_part5.npy', 'attention_masks_part6.npy']

input_ids_list = []
attention_masks_list = []

for input_ids_file, attention_masks_file in zip(input_ids_files, attention_masks_files):
    input_ids_part = np.load(input_ids_file, mmap_mode='r')
    attention_masks_part = np.load(attention_masks_file, mmap_mode='r')
    input_ids_list.append(input_ids_part)
    attention_masks_list.append(attention_masks_part)

input_ids = np.concatenate(input_ids_list, axis=0)
attention_masks = np.concatenate(attention_masks_list, axis=0)
# 建立模型
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
# 編譯模型
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


test_input_ids = []
test_attention_masks = []

for entry in tqdm(test_data):
    claim = entry['claim']
    encoded = tokenizer.encode_plus()
    test_input_ids.append(encoded['input_ids'])
    test_attention_masks.append(encoded['attention_mask'])

test_input_ids = tf.concat(test_input_ids, axis=0)
test_attention_masks = tf.concat(test_attention_masks, axis=0)

predictions = loaded_model.predict(
    x={'input_ids': test_input_ids, 'attention_mask': test_attention_masks}
)

```
## References
- Mitchell A. Gordon, 2019, [All The Ways You Can Compress BERT](https://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)
- Stickland, et al., ICML’19
- Houlsby, et al., ICML’19



