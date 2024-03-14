# NCCU_NLP
This repository contains my assignments and code for the Natural Language Processing course at NCCU.

## Folder Description
* Automated Fact Retrieval and Verification System:    
The course provides a fact database and a claim. Students are required to develop an automated fact retrieval and verification system to validate the truthfulness of 	the claim. If the claim can be "supported" or "opposed" by facts, the system must also provide evidence sentences by retrieving articles from the database.   
* homework:   
Classroom assignments include Chinese_word_segmentation, Naive_Bayes_Classifier, Stance_Detection.	

## Project Description
[brief report](Automated%20Fact%20Retrieval%20and%20Verification%20System/brief_report.pdf)


## Example
```python
from transformers import BertTokenizer, TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
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



