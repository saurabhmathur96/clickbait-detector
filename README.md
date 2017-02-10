# Clickbait Detector

Detects clickbait headlines using deep learning.

Find the Chrome Extension [here](https://chrome.google.com/webstore/detail/this-is-clickbait/ppklhdlfnadnlnllnenceabhldpnafjm) ( built by [rahulkapoor90](https://github.com/rahulkapoor90/This-is-Clickbait) )

## Requirements
- Python 2.7.12
- Keras 1.2.1
- Tensorflow 0.12.1
- Numpy 1.11.1
- NLTK 3.2.1

## Getting Started
1. Install a virtualenv in the project directory

       virtualenv venv

2. Activate the virtualenv
    - On Windows:

          cd venv/Scripts
          activate
      
    - On Linux
    
          source venv/bin/activate

3. Install the requirements

        pip install -r requirements.txt
        
4. Try it out!
    Try running one of the [examples](#examples).

## Accuracy
Training Accuracy after 25 epochs = 93.8 % (loss = 0.1484)

Validation Accuracy after 25 epochs = 90.15 % (loss = 0.2670)

## Examples

```
$ python src/detect.py "Novak Djokovic stunned as Australian Open title defence ends against Denis Istomin"
Using TensorFlow backend.
headline is 0.33 % clickbaity
```

```
$ python src/detect.py "Just 22 Cute Animal Pictures You Need Right Now"
Using TensorFlow backend.
headline is 85.38 % clickbaity
```

```
$ python src/detect.py " 15 Beautifully Created Doors You Need To See Before You Die. The One In Soho Blew Me Away"
Using TensorFlow backend.
headline is 52.29 % clickbaity
```

```
$ python src/detect.py "French presidential candidate Emmanuel Macrons anti-system angle is a sham | Philippe Marlire"
Using TensorFlow backend.
headline is 0.05 % clickbaity
```

## Model Summary
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
embedding_1 (Embedding)          (None, 20, 30)        195000      embedding_input_1[0][0]          
____________________________________________________________________________________________________
convolution1d_1 (Convolution1D)  (None, 19, 32)        1952        embedding_1[0][0]                
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 19, 32)        128         convolution1d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 19, 32)        0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution1d_2 (Convolution1D)  (None, 18, 32)        2080        activation_1[0][0]               
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 18, 32)        128         convolution1d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 18, 32)        0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
convolution1d_3 (Convolution1D)  (None, 17, 32)        2080        activation_2[0][0]               
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 17, 32)        128         convolution1d_3[0][0]            
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 17, 32)        0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
maxpooling1d_1 (MaxPooling1D)    (None, 1, 32)         0           activation_3[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 32)            0           maxpooling1d_1[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             33          flatten_1[0][0]                  
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 1)             4           dense_1[0][0]                    
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 1)             0           batchnormalization_4[0][0]       
====================================================================================================
Total params: 201,533
Trainable params: 201,339
Non-trainable params: 194
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
I used Stanford's Glove Pretrained Embeddings PCA-ed to 30 dimensions. This sped up the
training.


## Improving accuracy
To improve Accuracy, 
- Increase Embedding layer dimension (Currently it is 30) - `src/preprocess_embeddings.py`
- Use more data
- Increase vocabulary size - `src/preprocess_text.py`
- Increase maximum sequence length - `src/train.py`
- Do better data cleaning
