# Clickbait Detector

Detects clickbait headlines using deep learning.

## Accuracy
Training Accuracy after 25 epochs = 92.3 %

Validation Accuracy after 25 epochs = 88.6 %

## Examples

```
$ python bin/detect.py "Novak Djokovic stunned as Australian Open title defence ends against Denis Istomin"
Using TensorFlow backend.
headline is 5.17 % clickbaity
```

```
$ python bin/detect.py "Just 22 Cute Animal Pictures You Need Right Now"
Using TensorFlow backend.
headline is 91.31 % clickbaity
```

```
$ python bin/detect.py " 15 Beautifully Created Doors You Need To See Before You Die. The One In Soho Blew Me Away"
Using TensorFlow backend.
headline is 80.2 % clickbaity
```

```
$ python bin/detect.py "French presidential candidate Emmanuel Macrons anti-system angle is a sham | Philippe Marlire"
Using TensorFlow backend.
headline is 0.07 % clickbaity
```

## Model Summary
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
embedding_1 (Embedding)          (None, 20, 30)        195000      embedding_input_1[0][0]          
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 20, 30)        120         embedding_1[0][0]                
____________________________________________________________________________________________________
convolution1d_1 (Convolution1D)  (None, 19, 16)        976         batchnormalization_1[0][0]       
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 19, 16)        0           convolution1d_1[0][0]            
____________________________________________________________________________________________________
maxpooling1d_1 (MaxPooling1D)    (None, 9, 16)         0           activation_1[0][0]               
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 9, 16)         64          maxpooling1d_1[0][0]             
____________________________________________________________________________________________________
convolution1d_2 (Convolution1D)  (None, 8, 16)         528         batchnormalization_2[0][0]       
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 8, 16)         0           convolution1d_2[0][0]            
____________________________________________________________________________________________________
maxpooling1d_2 (MaxPooling1D)    (None, 1, 16)         0           activation_2[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 16)            0           maxpooling1d_2[0][0]             
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 16)            64          flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             17          batchnormalization_3[0][0]       
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 1)             0           dense_1[0][0]                    
====================================================================================================
Total params: 196,769
Trainable params: 196,645
Non-trainable params: 124
____________________________________________________________________________________________________
```


## Data
The dataset consists of about 12,000 headlines half of which are clickbait.
The clickbait headlines were fetched from BuzzFeed, NewsWeek, The Times of India and,
The Huffington Post.
The genuine/non-clickbait headlines were fetched from The Hindu, The Guardian, The Economist,
TechCrunch, The wall street journal, National Geographic and, The Indian Express.

Some of the data was from 
[peterldowns's clickbait-classifier repository](https://github.com/peterldowns/clickbait-classifier.git)


## Pretrained Embeddings
I used Stanford's Glove Pretrained Embeddings PCA-ed to 30 dimensions. This speeded up the
training.


## Improving accuracy
To improve Accuracy, 
- Increase Embedding layer dimension (Currently it is 30) - `bin/preprocess_embeddings.py`
- Use more data
- Increase vocabulary size - `bin/preprocess_text.py`
- Increase maximum sequence length - `bin/train.py`
- Do better data cleaning
