import pattern
import pattern.text
from pattern.text.en import sentiment
from pattern.text.en import parse

print(sentiment("The course of true love never did run smooth."))
print(sentiment("I love you."))
print(sentiment("Kolkata is very beautiful."))

pprint(parse('I drove my car to the hospital yesterday', relations=True, lemmata=True))

text = "Paris is the capital of France"
sent = parse(text, lemmata=True)
sent = Sentence(sent)

print(modality(sent))