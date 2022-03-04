from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
# Bottom ssl is workaround for broken script on punkt donwloadm which returns a loading ssl error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
# End of error workaround

stemmer = PorterStemmer()
# Imports needed from nltk

# Our Tokenizer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stemming Function


def stem(word):
    return stemmer.stem(word.lower())

# Bag of Words Function


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


if __name__ == '__main__':

    # TODO: Test our function with the below sentence to visualize Tokenization.
    # TODO CONT: What is the purpose of tokenizing our text?
    """
    Tokenization is to split a string into meaningful units, such as words, punctuation characters, numbers.
    """
    # Testing our Tokenizer
    test_sentence = "I will not live in peace until I find the Avatar!"
    tockens = tokenize(test_sentence)
    print("---------Test Tokenization function---------")
    for token in tockens:
        print(token)

    # TODO: Test our Stemming function on the below words.
    # TODO CONT: How does stemming affect our data?
    """
    Stemmimg generate the root from of the words. It's a crude heuristic that chops of the ends off of words.
    """
    words = ["Organize", "organizes", "organizing", "disorganized"]
    print("---------Test Stemming function---------")
    for word in words:
        print(f'The root of {word} is {stem(word)}')

    # TODO: Implement the above Bag of Words function on the below sentence and words.
    # TODO (CONTINUED): What does the Bag of Words model do? Why would we use Bag of Words here instead of TF-IDF or Word2Vec?
    """
    We can't just pass in the sentence as it is to the neural net, 
    so we need to convert the pattern of strings to numbers that the netwrok can understand.
    A bag of words has all the words in a array and each position contains a 1 if the word is available in the sentence.
    In this case, we simply want all the unique words to figure out the sentence pattern,
    and feed it to the neural net to get the probabilities for diffirent classes.
    We don't need to know which word is the more important or the weight of it, so the bag of words would be suitable here.
    """
    print("---------Test Bag of Words function---------")
    sentence = ["I", "will", "now", "live", "in",
                "peace", "until", "I", "find", "the", "Avatar"]
    words = ["hi", "hello", "I", "you", "the",
             "bye", "in", "cool", "wild", "find"]
    print(f'Bog= {bag_of_words(sentence,words)}')
    print("--------------")
