# Retrieve Relevant Image to Given Text

## Objective
  This is a machine learning Project. In our present generation we can find number of images and captions in different platforms like social media platforms,movie posters,in advertisements etc.
Most of the times instead of observing the complete image clearly, we just read the caption and understand the complete matter involved in it.
So,At that time the relation between text and image must be more accurate. To find the relationship accurate between those we developed this project.

## Insights
  Here we find the relationship value between the image and text.That value is between 1 and -1. Based on that value we can predict they are closely related or not. Value 1 represents they are very closely related.
  0 represents they are not closely related but -1 value represents they are quite opposite.

## Metrics Used:
* TfidVectorizer: A metric used to convert a value from text format to numerical vector array format.
* Image: Used to convert image format to Vector array format
* Cosine Similarity: A metric used to find the relationship value between the two vectors.
