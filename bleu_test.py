from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

s1 = 'results for trust in recent immigrants are in line with expectations'
s2 = 'the results are new the and and and and new 0'

s3 = 'will the subsidized cook have learned the skills necessary to do the job'
s4 = 'will the cra will be the the the by the the 0 0'

t1 = ['this', 'is', 'a', 'test']
t2 = ['this', 'it', 'is', 'test']

ss1 = s1.split()
ss2 = s2.split()

ss3 = s3.split()
ss4 = s4.split()

print("------------------------")
print(bleu_score(t1,t2, weights=(1, 0, 0, 0)))

references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'it', 'a', 'test']]

score = corpus_bleu([ss1], [ss2], weights=(1, 0, 0, 0))
print(score)