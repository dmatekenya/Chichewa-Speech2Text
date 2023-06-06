# Automated Speech Recognition(ASR) for Chichewa

This repository shares results, code and activities for work whose goal is to contribute to improvements in ASR or speech to text technologies (e.g., NLP models)  for <span style="color:#3EACAD">Chichewa</span> which happens to be my native language and also the major language spoken in Malawi. In Zambia and Mozambique, Chichewa is also known as *Chinyanja*,  *Nyanja* or *Chewa*. At the time of writing of this document, over <span style="color:#3EACAD">21 million people</span> speak the language. The project has two main goals as follows: 

1. **Generate speech and other NLP datasets.** Contribute to generation of open source speech datasets for the language. Since quality of results from speech systems is inevitably connected to availability and quality of other NLP datasets in the language, the project is also interested in contributing to other NLP datasets such as those for translations. 
2. **Enhance ASR models.** Since Chichewa is a low resource language, the quality of speech to texts results from platforms such as Google is poor. As such, the goal here is to fine-tune these large language models with properly curated data to improve performance of the models.  Again, this applies to other downstream NLP tasks such as *machine translation*, *text summarization* and *question answering*. Furthermore, improving accessibility and usage of these models among locals in the target countries is another important objective.

The repository contains the following folders:

- **docs/**

    > This folder contains documents pertaining to the project and the repo itsself.
     
- **data/**
    > The datasets in this folder will not be uploaded to the repo as most of them are too large. However, finalized datasets will be shared through platforms such as Zenodo. 
- **src/**
    > All the Python and other code used is placed here. For example, this folder contains source code for the speech data collection app: LIG-Akuma
    
- **models/**
    > Any models available after tuning will be here  

- **notebooks/**
    > All notebook files are here. 
    
## Speech and other NLP Datasets
### Mozilla common voice
One of the objectives of this project is to get Chichewa on the  [Mozilla common voice](https://commonvoice.mozilla.org/en) platform and accelerate collection of speech data and consequently better ASR models for the language. To this end, the process to onboard the language has already been initiated. Currently, we are in the processl what Mozilla calls **language localization** which involves translation of text strings from English to Chichewa. Please go to [translate English to Chichewa to help with this process](https://pontoon.mozilla.org/ny/). 

### Zindi-Google NLP competition
Although I have had some interest in NLP as it relates to my native language, I never got myself to work on anything cooncrete until I came across the [google-asr-hack-series-africa-asr-data-challenge](https://zindi.africa/competitions/google-asr-hack-series-africa-asr-data-challenge) challenge on [Zindi](https://zindi.africa)[^1]. I participated in the challenge and came in first place and in the process published a [speech dataset for Chichewa
](https://doi.org/10.5281/zenodo.6595625) on Zenodo.

#### Other NLP datasets 
I have also found the following datasets on Chichewa:

1. [Chichewa news articles dataset](https://doi.org/10.5281/zenodo.4315018)
2. [Spoken Chichewa corpus.](https://doi.org/10.5281/zenodo.3731994) 


[^1]. A data science competition platform with focus on Africa challenges

## Enhancing NLP models
The ultimate goal of this work is to bring to Chichewa speakers high quality and accurate speech recognition systems as well as other langauge technology taks such as machine translation. Thus, once we have propery curated data, we need to use it to enhance existing models such as transformers to tune them so that they perform better on the langauge. The work on showing how large open models such as [Facebook wav2vec ](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)  can be fine-tuned to improve perfomance on Chichewa is still on-going and weill share results soon.

## Usage and contributing
There are several ways to contribute to this project as follows:
### Data contributions
For low resource langauge such as Chichewa, the main reason why they lag behind in NLP technologies is due to lack of data. As such, one of the biggest contribution you can contribute is to share your data as a native speaker of the language and/or contribute to data collection initiatives for non-speakers.

1. [Translate English to Chichewa to help](https://pontoon.mozilla.org/ny/) get the langauge on Mozilla common voice platform.
2. **Share audio clips in Chichewa.** If you have any audio clips in Chichewa, you can share them. Please use the contact email below to inquire how.
3. **Share Chichewa text corpora.** If you have Chichewa books, news articles, documents  or any other texts, please share them as these are useful for creating language models.lease use the contact email below to inquire how.
4.  **Share expertise.** If you are a Chichewa linguists who is passionate about language technologies, please contact me so that we can work together. 



### Technical contributions 
If you are a data scientist, machine learning engineer, computational linguist, linguist or any other person with interest in advancing NLP for low-resource languages or with specific interest in Chichewa, you can contribute code by the following:

1. **Benchmarking models.** Test performance of open source models, online services (e.g., Goog;e translate) for this language across different NLP tasks such as ASR, machine translation, text summarization and more
2. **Fine-tune models.** Use open source Chichewa data and any other data to stretch performance of open source models (e.g., BERT) on this language
3. **Any other activity/collaboration.** which may lead to advancement of NLP for Chichewa.

## Contacts
If you would like to get involved in this project, please contact me on the email below:

- **Dunstan Matekenya:** dmatekenya@gmail.com

