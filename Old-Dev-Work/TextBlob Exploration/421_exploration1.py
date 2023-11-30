from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import json
import os

#start https://textblob.readthedocs.io/en/dev/quickstart.html#tokenization
#classifiers https://textblob.readthedocs.io/en/dev/classifiers.html#classifiers
#different classifiers https://textblob.readthedocs.io/en/dev/api_reference.html#api-classifiers

train = [
     ('I love this sandwich.', 'pos'),
     ('this is an amazing place!', 'pos'),
     ('I feel very good about these beers.', 'pos'),
     ('this is my best work.', 'pos'),
     ("what an awesome view", 'pos'),
     ('I do not like this restaurant', 'neg'),
     ('I am tired of this stuff.', 'neg'),
     ("I can't deal with this", 'neg'),
     ('he is my sworn enemy!', 'neg'),
     ('my boss is horrible.', 'neg')
 ]
test = [
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')
 ]

#Polarity score and subjectivity score of a sentence

def sentiment_analysis(sentence):
    def get_subjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    
    def get_polarity(text):
        return TextBlob(text).sentiment.polarity
    
    sub = get_subjectivity(sentence)
    pol = get_polarity(sentence)
    #sentence['TextBlob_Subjectivity'] = sentence['sentence'].apply(get_subjectivity)
    #sentence['TextBlob_Polarity'] = sentence['sentence'].apply(get_polarity)
    return [pol,sub]
    #return sentence

def sentiment_classify(sentence):
    score = sentiment_analysis(sentence)
    if score[0] >= 0.03:
        return "pos"
    elif score[0] <= -0.03:
        return "neg"
    else:
        return "neu"

def compare(data, annotations): #compares how many we correctly classify
    num_correct = 0
    total = 0
    sentence_list = []
    for i in data['body-paragraphs']:
        for j in i:
            sentence_list.append(j)
    for phrase in annotations["phrase-level-annotations"]:
        if(phrase["id"] == "title"):
            if (sentiment_classify(data["title"]) == phrase["polarity"]):
                num_correct += 1
            total += 1
        else:
            sentence_id = int(phrase["id"][1:])
            if (sentiment_classify(sentence_list[sentence_id]) == phrase["polarity"]):
                num_correct += 1
            total += 1
    return total, num_correct

def tokenize_words(text):
    return TextBlob(text).words

#sentences and TextBlobs are equivalent in properties/methods
def tokenize_sentences(text):
    return TextBlob(text).sentences

def parse_text(text):
    return TextBlob(text).parse()

def end_word_extractor(document):
     tokens = document.split()
     first_word, last_word = tokens[0], tokens[-1]
     feats = {}
     feats["first({0})".format(first_word)] = True
     feats["last({0})".format(last_word)] = False
     return feats

###START OF TESTING CODE

classifier1 = NaiveBayesClassifier(train)
classifier2 = NaiveBayesClassifier(train, feature_extractor=end_word_extractor)

wiki = TextBlob("Python is a high-level, general-purpose programming language.")
print(wiki.tags)

test_sent1 = "I hate the things that Trump has done over the years."
print(sentiment_analysis(test_sent1)) #polarity,subjectivity score
#print(classify(test_sent1))
print("classifier1: " + classifier1.classify(test_sent1)) #pos,neutral,neg based on training data
print("classifier2: " + classifier2.classify(test_sent1)) #classify with added feature extractor

test_sent2 = "In an Epic Battle of Tanks, Russia Was Routed, Repeating Earlier Mistakes"
print(sentiment_analysis(test_sent2))
print(classifier1.classify(test_sent2))

test_sent3 = "Ukraine war live updates: Russian mercenary boss says ‘fierce resistance’ seen in Bakhmut; Kyiv says its fighters are under ‘insane pressure’"
print(sentiment_analysis(test_sent3))
print(classifier1.classify(test_sent3))

print("classifier 1 accuracy: %.3f" % classifier1.accuracy(test)) #accuracy of classifier given test data
print("classifier 2 accuracy: %.3f" % classifier2.accuracy(test))

file_list = []

for i in range(10):
    file_i = os.listdir("BASILdata/articles/201" + str(i))
    for file_name in file_i:
        file = open("BASILdata/articles/201" + str(i) + "/" + file_name, encoding="utf8")
        json_file = json.load(file)
        file_list.append(json_file)
        
annotation_file_list = []
for i in range(10):
    file_i = os.listdir("BASILdata/annotations/201" + str(i))
    for file_name in file_i:
        file = open("BASILdata/annotations/201" + str(i) + "/" + file_name, encoding="utf8")
        json_file = json.load(file)
        annotation_file_list.append(json_file)
total_correct = 0
total = 0
for i in range(len(file_list)):
    data = file_list[i]
    annotations = annotation_file_list[i]
    a, b = compare(data, annotations)
    total += a
    total_correct += b
accuracy = total_correct/total
print("BASIL correct: %.3f , BASIL total: %.3f" %(total_correct, total))
print(accuracy)