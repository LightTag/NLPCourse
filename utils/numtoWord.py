import numpy as np
numMap = {

    0: '',
    1: 'eins',
    2: 'zwei',
    3: 'drei',
    4: 'vier',
    5: 'fünf',
    6: 'sechs',
    7: 'sieben',
    8: 'acht',
    9: 'neun',
    10: 'zehn',
    11: 'elf',
    12: 'zwölf',
    20: 'zwanzig',
    30: 'dreiβig',
    40: 'vierzig',
    50: 'fünfzig',
    60: 'sechzig',
    70: 'siebzig',
    80: 'achtzig',
    90: 'neunzig',
    None: '',
    10 ** 2: 'hundert',
    10 ** 3: 'tausend',
    10 ** 6: ' Million ',
    10 ** 9: 'milliarde',
    10 ** 12: 'billion',
    10 ** 15: 'billiarde',
    10 ** 18: 'trillion',

}

def makeCharVocab():
    chars = set(''.join(numMap.values()))
    vocab = {
        '_PAD_':0,
        '_START_':1,
        '_END_':2
    }
    for num,char in enumerate(chars):
        vocab[char] = num+3
    return vocab

vocab = makeCharVocab()
reverseVocab = {val:key for key,val in vocab.items()}



def handleRightPair(b, c, only=False):
    # handles tens places
    B, C = map(int, [b, c])
    if B == 0:
        # if its a ones
        result = numMap[C]
    elif B == 1 and C < 3 and only:
        # eleven or twelve
        result = numMap[10 * B + C]
    elif B == 1:
        # if its 10
        result = numMap[C] + numMap[10 * B]
    else:

        result = numMap[C] + 'und' + numMap[10 * B]

    if not (only and B == 0):
        # if its just 1
        result = result.replace('eins', 'ein')
    return result


def handleTriplet(triplet, only=False):
    a, b, c = list(triplet)
    A, B, C = map(int, [a, b, c])
    tens = handleRightPair(b, c, only=only)

    if A == 0:
        result = tens
    elif A == 1:
        result = numMap[A][:-1] + 'hundret' + tens
    else:
        result = numMap[A] + 'hundret' + tens
    return result


from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def handleNumber(num):
    '''
    Givena  number, returns it spelled out as a word
    8535348 ==>   'acht Million fünfhundretfünfunddreiβigtausenddreihundretachtundvierzig'
    :param num:
    :return:
    '''
    only = num < 1000
    size = None
    snum = str(num)
    rnum = snum[::-1]
    triplets = []
    place = 0
    while place < len(snum):
        triplet = list(rnum[place:place + 3].ljust(3, '0'))[::-1]
        triplets.append(triplet)
        place += 3

    result = ''
    for triplet in triplets:
        result = handleTriplet(triplet, only) + numMap[size] + result
        if size == None:
            size = 1000
        else:
            size *= 1000
    return result





def createNum2WordDict(size,high=(10 **18) - 1):
    result = {}
    for num in np.random.randint(0,high,size):
        result[num] = handleNumber(num)
    return result
