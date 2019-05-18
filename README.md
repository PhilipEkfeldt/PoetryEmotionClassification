# Multi-Label Emotion Classification in English Poetry using Song Lyricsand a Dual Attention Transfer Mechanism

In this paper, we attempt to perform multi-label emotion classification on English poetry. This domain presents a number of challenges, chief among which are the apparent lack of annotated poetry data and the subjectivity involved in emotion annotation. In an attempt to determine the degree to which song lyrics are similar to poetry specific to the ways in which they express or evoke emotions, we employ a dual attention transfer mechanism and augment the learning of emotion from poetry using sentiment learned from song lyrics.

***

As online platforms where users share their personal opinions grow in popularity, emotion detection has the potential to be a powerful tool in domains such as social media, political science, marketing, and human-computer interaction. Our intention is to learn and yield accurate emotion classifications as understanding people's emotions can help improve the performance of personalized recommendations and services. Previous literature predominantly makes use of Twitter data. Instead, we choose to use emotion-annotated poetry data, as poetry represents a domain of textual data that has significantly more emotional nuance. 

Deep learning methods require large annotated data sets, and we were hence presented with the problem of finding sufficient emotion-annotated poetry data. Extensive efforts to find such a data set were unsuccessful. Subsequently, the annotation task was outsourced to Amazon Mechanical Turk.  

Using a bi-directional LSTM model to establish a baseline, we proceed to implement a dual attention transfer mechanism (Yu et al., 2018). We address the lack of emotion-annotated poetry data by using sentiment-annotated song lyrics data to augment the effectiveness of the emotion classification model.
