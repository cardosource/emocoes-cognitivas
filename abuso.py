from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import FreqDist, NaiveBayesClassifier
from nltk.classify import apply_features
# import nltk
# nltk.download('rslp')
# nltk.download('stopwords')

sujeito = 'Eu vi teu perfil e fiquei interessado'

frases_padrao = [
    ('vi observei olhei passei olhando indicação sujestão comun passando interessado afim fiquei queria quero perfil conta', 'suspeitar'),

    ('amigo amiga colega conhecido conhecida primo prima alguem fulano pessoa ciclano irmão irmã tio tia filho filha', 'duvidar'),

    ('contato número adicionar add passar pode porfavor favor poderia inportaria gentileza falar conversar', 'medo'),
]




# remove preposição artigo pronome

def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwords.words('portuguese')]
        frases += [(semstop, emocao)]
    return frases


def encontrarpalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras


palavras = encontrarpalavras(removestopwords(frases_padrao))


def buscafrequencia(palavras):
    palavras = FreqDist(palavras)
    return palavras



def encontrarpalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq


palavrasunicas = encontrarpalavrasunicas(buscafrequencia(palavras))


def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavrasunicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas


classificador = NaiveBayesClassifier.train(apply_features(extratorpalavras, removestopwords(frases_padrao)))

testestemming = []
stemmer = RSLPStemmer()
for (palavrastreinamento) in sujeito.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))

print('individuo: %s -  <reação da vitima = %s> ' % (sujeito, classificador.classify(extratorpalavras(testestemming))))
