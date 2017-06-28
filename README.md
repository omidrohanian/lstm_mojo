A simple LSTM network for sequence labelling of web-scraped text 

We first extract some text from semanticlaly related and pre-selected articles from Wikipedia, preprocess and run it through an 
NLTK POS tagger. Then we use these tags as labels and redefine the problem as a sequence labelling task using an [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) network.


Prerequisites:

sklearn

[wikipedia ](https://pypi.python.org/pypi/wikipedia/)

tensorflow

keras 

numpy 

nltk 
