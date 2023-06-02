## Training the Models For Sentiment Analysis Using Deep Learning
### How to use
1. Download the Sentiment140 dataset from the given link `https://www.kaggle.com/datasets/kazanova/sentiment140`
2. Open `data_preprocess.ipynb` to preprocess the dataset to be used and save embedding model using gensim library from this same notebook.
3. Open `model_training.ipynb` to train the model using tensorflow and saving the deep learning model.
4. Open `model_predicting` to load the trained model and test on your own text.
5. The backend server created using fastapi for this project can be found at `https://github.com/niranjanblank/SentimentAnalysisBackend`
6. The frontend created using react.js for this project can be found at `https://github.com/niranjanblank/SentimentAnalysisFrontend`