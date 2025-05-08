# Common-Crawl-News-Article-Classification

At the time of this project, ARCYBER is standing up Theater Informataion Advantage Detachments (TIAD). This process fits into our broader capstone project, which aims to equip TIAD analysts with information advantage dashboards leveraging open source news datasets. Various machine learning enrichments are done to this news data to provide analysts with a better understanding of what is occuring in the event space. We aim to classify articles into one of 12 categories : Health, business, politics, national security, cyber, sports, weather, crime, entertainment, culture, advertising, and technology. We employ both multiclass (predicting only one topic per article) and multilabel (predicting multiple topics per article). 

## Training Data Labeling
The first process began with labeling our training data with an LLM. In this project, Gemma 3 was utilized to classify articles into the various topics mentioned above. Our training data consisted of roughly 492,000 news articles from Common Crawl. Once we had this labeled data, we needed to balance the topics for trianing. Politics was the dominant topic, with roughly 91,000 articles being tagged as political articles. Science was our minority class, with only 2864 articles. We experimented between up and down sampling, with the only significant difference being training time of the model. No noticable differences in accuracy were assessed. 

## Embedding
We employed various embedding types between Robustly Optimized BERT embeddings (roBERTa), Language-agnostic BERT sentence embeddings (LaBSE), and Bidirectional Encoder Representations for Transformers (BERT). While roBERTa embeddings have a higher dimensional semantic embedding space, the model performance would deteriorate on less prominent languages. Since Common Crawl represents nearly 92 different languages, we opted to use LaBSE embeddings. 

## Model Training
We trained two networks, one for multiclass, and one for multilabel prediction. Both models were trained for roughly 40 epochs, taking about an hour to train on the downsampled data. Both models were roughly 8 million parameters

## Performance
Our multiclass prediction model achieved approximately 70% accuracy on a test set of around 13,000 articles. However, many articles naturally span multiple topics, which negatively impacted performance under the single-label constraint. By transitioning to a multilabel model, we improved accuracy to about 80% using softmax outputs with a 25% threshold. For instance, if the model predicted "tech," "cyber," and "national security," and the LLM-labeled ground truth included "cyber," the prediction was considered correct.

## Dependencies
To run this script on the wire, downgrade Tensorflow to version 2.11 for compatability with the GPU's. 
