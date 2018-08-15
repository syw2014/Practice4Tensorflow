#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: phrase.py
# Date: 18-5-30 上午11:13


import codecs


corpus = []
with codecs.open("../data/test_pharse.txt", encoding='utf-8') as f:
    for line in f.readlines():
        corpus += line.split('/')

# from gensim.models import Phrases
# documents = ["the mayor of new york was there", "machine learning can be useful sometimes","new york mayor was present"]
#
# sentence_stream = [doc.split(" ") for doc in documents]
# bigram = Phrases(sentence_stream, min_count=1, threshold=2)
# sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
# print(bigram[sent])

import jieba
import nltk
from nltk.collocations import *
train_corpus = "昨日/2018LPL/春季/赛季/赛首周/结束/BLG/领先/情况下/比分/赢得/比赛/进入/到此/代表/LPL/参加/2018/亚洲/对抗赛/四支" \
               "/战队/产生/赛区/IG/RNG/西部/赛区/EDG/IG/获得/赛事/参赛/资格/亚洲/对抗赛/LPL/LMS/LCK/进行/韩国/LCK/联赛/KINGZONE" \
               "/AFREECA/KT/TELECOM/T1/出战/参加过/赛事/SAMSUNG/GALAXY/MVP/战队/未能/入围/LCK/春季赛/冠亚军/进入/赛事/希望/能够" \
               "/保留/风格/LMS/联赛/FLASH/WOLVES/REX/MAD/TEAM/MACHI/SPORTS/出战/LMS/变化/FLASH/WOLVES/闪电狼/掌控/雷电/宿敌" \
               "/SPORTS/CLUB/TEAM/2018/LMS/春季赛/表现/未能/进入/季后赛/香港/英皇娱乐/组建/REX/战队/TOYZ/教练/带领/闪电狼/形成" \
               "/争霸/格局/实力/不容/小觑/昨日/LPL/春季赛/诞生/RIOT/中国/负责/电子竞技/选手/管理/微博/透露/亚洲/对抗赛/举办地/" \
               "辽宁大连/大连/举办/LOL/社区/相关/主题/大佬/暗示/得到/拳头/官方/工作人员/确认/尚属/亚洲/对抗赛/世界/总决赛/区域/" \
               "交流/意义/赛事/看点/LPL/蝉联冠军/心态/进入/世界/总决赛"
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

finder = BigramCollocationFinder.from_words(train_corpus.split('/'))
finder.apply_word_filter(lambda w: w.lower() in [',', '.', '，', '。'])
res1 = finder.nbest(bigram_measures.pmi, 10)
for x in res1:
    print(x[0] + " " + x[1])
print("==============")
finder = TrigramCollocationFinder.from_words(train_corpus.split('/'))
finder.apply_word_filter(lambda w: w.lower() in [',', '.', '，', '。'])
res = finder.nbest(trigram_measures.pmi, 15)
for x in res:
    print(x[0] + " " + x[1] + " " + x[-1])

#用gensim+jieba发现连词
# import jieba
# import gensim
#
#
# mddesc = ['测试数据库','用户支付表','支付金额','支付用户']
# train_corpus = []
# for desc in mddesc:
#     train_corpus.append("/".join(jieba.cut(desc)).split("/"))
#     train_corpus.append("/".join(jieba.cut(desc)).split("/"))
#
#
# #set the params(min_count, threshold) carefully when you use small corpus.
# phrases = gensim.models.phrases.Phrases(train_corpus, min_count = 1, threshold=0.1)
# bigram = gensim.models.phrases.Phraser(phrases)
# input = "从用户支付表中选择支付金额大于5的用户。"
# inputarr = "/".join(jieba.cut(input)).split("/")
# repl = [s.replace("_","") for s in bigram[inputarr]]
# for x in bigram[inputarr]:
#     print(x)