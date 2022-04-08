import os
import re
import string
import gensim as gensim
import nltk
from nltk.collocations import *
import pyLDAvis as pyLDAvis
from gensim import corpora
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk import FreqDist
from nltk.corpus import webtext
from nltk.corpus import stopwords
import json




with open("data2.json", "r") as f:
    data2 = json.load(f)

#bygrams
text = "Eşe karşı basit yaralama suçundan sanık ...'ın, 5237 sayılı Türk Ceza Kanunu'nun 86/2, 86/3-a, 62/1 ve 52/2. maddeleri uyarınca 3.000,00 Türk Lirası adlî para cezası ile cezalandırılmasına, 5271 sayılı Ceza Muhakemesi Kanunu'nun 231/5. maddesi gereğince hükmün açıklanmasının geri bırakılmasına dair Kırıkkale 4. Asliye Ceza Mahkemesinin 07.11.2019 tarihli ve 2018/535 Esas, 2019/948 Karar sayılı kararına karşı yapılan itirazın kabulü ile anılan kararın kaldırılmasına ilişkin mercii Kırıkkale 1. Ağır Ceza Mahkemesinin 05.02.2020 tarihli ve 2020/139 değişik iş sayılı kararına karşı Adalet Bakanlığının 17.11.2020 tarihli ve 2020/9153 sayılı yazısıyla kanun yararına bozma isteminde bulunulduğundan bu işe ait dava dosyası Yargıtay Cumhuriyet Başsavcılığının 15.12.2020 tarihli ve 2020/105078 sayılı tebliğnamesi ile Dairemize gönderilmekle incelendi.  Mezkur ihbarnamede;  5271 sayılı Kanun’un 231/8. maddesinde yer alan “Denetim süresi içinde, kişi hakkında kasıtlı bir suç nedeniyle bir daha hükmün açıklanmasının geri bırakılmasına karar verilemez.” şeklindeki düzenleme gereğince, denetim süresinde bir kez daha suç işlenmesi halinde ikinci suçtan dolayı kurulan hükmün tekrar açıklanmasının geri bırakılamayacağı nazara alındığında, sanık hakkında 06.06.2017 tarihinde işlemiş olduğu suçtan dolayı Kırıkkale 3. Asliye Ceza Mahkemesinin 10.05.2018 tarihli ve 2017/523 Esas, 2018/315 sayılı kararıyla verilen hükmün açıklanmasının geri bırakılmasına dair kararın 08.10.2018 tarihinde kesinleştiği ve denetim süresinin de bu tarih itibariyle başladığı, yargılama konusu suçun ise önceki kararın kesinleşmesinden önce 21.05.2018 tarihinde işlendiği cihetle, denetim süresi içerisinde işlenmiş bir suçtan bahsedilemeyeceği gözetilmeden, itirazın reddi yerine yazılı şekilde kabulüne karar verilmesinde isabet görülmediğinden bahisle, 5271 sayılı CMK'nin 309. maddesi gereğince anılan kararların bozulması lüzumunun ihbar olunduğu anlaşıldı.  Gereği görüşülüp düşünüldü:  5271 sayılı CMK’nin 231. maddesinde düzenlenen “hükmün açıklanmasının geri bırakılması” müessesesinin uygulanabilmesi için öncelikle,  - Sanık hakkında kurulan mahkûmiyet hükmünde, hükmolunan cezanın iki yıl veya daha az süreli hapis veya adli para cezasından ibaret olması,  - Suçun CMK’nın 231. maddesinin 14. fıkrasında yazılı suçlardan olmaması,  - Sanığın daha önce kasıtlı bir suçtan mahkûm olmamış bulunması,  - Sanığın hükmün açıklanmasının geri bırakılmasına itirazının bulunmaması,  Suçun işlenmesiyle mağdurun veya kamunun uğradığı zararın, aynen iade, suçtan önceki hale getirme veya tamamen giderilmesine ilişkin koşulların birlikte gerçekleşmesi gerekmektedir.  Ayrıca, bahsi geçen maddenin 8. fıkrasında; \"Hükmün açıklanmasının geri bırakılması kararının verilmesi halinde sanık, beş yıl süreyle denetim süresine tâbi tutulur. (Ek cümle: 18/06/2014-6545 S.K./72. md) Denetim süresi içinde, kişi hakkında kasıtlı bir suç nedeniyle bir daha hükmün açıklanmasının geri bırakılmasına karar verilemez.” hükmü yer almaktadır.  CMK’nin 231/8. maddesine ilişkin 6545 sayılı Kanun’un 72. maddesinin gerekçesinde de bu durum; “Maddeyle, Ceza Muhakemesi Kanunu'nun 231. maddesinin sekizinci fıkrasında değişiklik yapmak suretiyle, hükmün açıklanmasının geri bırakılmasına karar verilmesi halinde sanığın tabi tutulacağı denetim süresi içinde sanık hakkında bir daha hükmün açıklanmasının geri bırakılmasına karar verilemeyeceği düzenlenmektedir. Söz konusu maddenin uygulanmasında, hükmün açıklanmasının geri bırakılması kararı verilen sanıklar hakkında işledikleri diğer suçlardan dolayı da birçok kez hükmün açıklanmasının geri bırakılması kararı verildiği görülmektedir. Yapılması öngörülen değişiklikle, bu uygulamaya son verilmesi ve denetim süresi içinde sanık hakkında bir daha hükmün açıklanmasının geri bırakılmasına karar verilememesi amaçlanmaktadır. Kişinin işlediği ikinci suçun denetim süresi içinde işlenip işlenmediğinin önemi bulunmamaktadır. Daha önceden işlenen suçlar bakımından da bu yasak uygulanacaktır.” şeklinde ifade edilmiştir.  Buna göre 6545 sayılı Kanun'un yürürlüğe girdiği tarihten sonra işlenen suçlar için, hakkında daha önce hükmün açıklanmasının geri bırakılması kararı bulunan sanıklarla ilgili bir daha hükmün açıklanmasının geri bırakılması kararı verilemeyecektir.  İnceleme konusu somut olayda; mahkemece sanık ...’ın kasten basit yaralama suçundan adli para cezası ile cezalandırılmasına ve hükmün açıklanmasının geri bırakılmasına karar verilmiştir. Bu karara karşı yapılan itiraz merciince kabul edilerek sanık hakkındaki hükmün açıklanmasının geri bırakılmasına dair karar kaldırılmıştır.  Sanığın adli sicil kaydında yer alan kasten basit yaralama suçundan verilen Kırıkkale 3. Asliye Ceza Mahkemesinin 10.05.2018 tarihli ve 2017/523 Esas, 2018/315 Karar sayılı hükmün açıklanmasının geri bırakılması kararının 08.10.2018 tarihinde kesinleşmesi üzerine bu suç yönünden sanık hakkında denetim süresi başlamıştır. Böylece CMK’nin 231/8. maddesindeki; “Denetim süresi içinde, kişi hakkında kasıtlı bir suç nedeniyle bir daha hükmün açıklanmasının geri bırakılmasına karar verilemez” şeklindeki düzenleme gereğince inceleme konusu kasten basit yaralama suçu yönünden sanık hakkında hükmün açıklanmasının geri bırakılmasına karar verilemeyecektir. Sanığın inceleme konusu kasten basit yaralama suçunu adli sicil kaydındaki diğer kasten basit yaralama suçundan verilen hükmün açıklanmasının geri bırakılması kararının kesinleşme tarihinden önceki bir tarihte gerçekleştirmiş olmasının önemi bulunmamaktadır. Zira sanığın bu suçtan verilen hükmün açıklanmasının geri bırakılması kararının denetim süresi 08.10.2018 tarihinde başlamış ve incelenen kasten basit yaralama suçundan mahkemece 07.11.2019 tarihinde karar verilmiştir. 08.10.2018 tarihinden sonra sanık hakkında kasıtlı bir suçtan yeni bir hükmün açıklanmasının geri bırakılmasına karar verilmesi mümkün bulunmamaktadır.  Böylece, sanık hakkında mahkemece hükmün açıklanmasının geri bırakılmasına dair karara yönelik itirazın merciince kabul edilerek kaldırılmasına karar verilmesinde isabetsizlik görülmemiştir.  Açıklanan bu nedenlerle, Adalet Bakanlığının kanun yararına bozma isteyen yazısına dayanan tebliğnamede ileri sürülen düşünce yerinde görülmeyerek kanun yararına bozma talebinin REDDİNE, dosyanın mahalline gönderilmek üzere Yargıtay Cumhuriyet Başsavcılığına TEVDİİNE, 04.01.2021 gününde oybirliği ile karar verildi. (¤¤)"
wordlist = text.split()
bigram_fd = nltk.FreqDist(nltk.bigrams(wordlist))
bigram_measures = nltk.collocations.BigramAssocMeasures()
words = [w.lower() for w in text]
bcf = BigramCollocationFinder.from_words(words)
bcf = BigramCollocationFinder.from_words(text)
Tokens = nltk.word_tokenize(text)
output = tuple(nltk.bigrams(Tokens))

finder = BigramCollocationFinder.from_words(text)
finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
finder = BigramCollocationFinder.from_words(nltk.corpus.brown.tagged_words('ca01', tagset='universal'))
finder = BigramCollocationFinder.from_words(Tokens)


filter_stops = lambda w: len(w) < 3 or w in stop_words
bcf.apply_word_filter(filter_stops)
word_in_text = word_tokenize(text)
stop_words = set(stopwords.words("turkish"))
filtered_list = []
for word in word_in_text:
    if word.casefold() not in stop_words:
        filtered_list.append(word)
word_filter = lambda *w: 'criminal' not in w
filtered_list = [word for word in word_in_text if word.casefold() not in stop_words]
finder = BigramCollocationFinder.from_words(filtered_list)
finder.apply_freq_filter(3)
finder.apply_ngram_filter(word_filter)
bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)
text_finder = BigramCollocationFinder.from_words(text)
text_scored = text_finder.score_ngrams(bigram_measures.raw_freq)
word_in_text = word_tokenize(text)
lemmatizer = WordNetLemmatizer()
nltk.pos_tag(word_in_text)
sent_tokenize(text)
lotr_pos_tags = nltk.pos_tag(word_in_text)
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(lotr_pos_tags)
#tree.draw()
frequency_distribution = FreqDist(text)
frequency_distribution.most_common(15)
frequency_distribution = FreqDist(filtered_list)

def cleaning(text):
    try:
        text = text.encode('utf-8','ignore').decode('utf-8')
    except:
        text = text.encode('ascii','ignore')
        text = re.sub('\\S+@\\S+|@\\S+', '', text)
        text = re.sub('\\n', '', text)
        text = re.sub('\\s', '', text)
        text = re.sub('[-|#()"":/*"]', '', text)
        #print(text)
    remove = string.punctuation
    remove = remove.replace(",", "")
    pattern = r"[{}]".format(remove)
    re.sub(pattern, "", text)
    text = re.sub('(.)\\1\\1+', '\\1', text)
    text = text.replace(",", " ,")
    return text

lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
new_text = nltk.Text(lemmatized_words)
new_text.collocations()
#sorted(bigram for bigram, score in scored)
bigram_fd.most_common()
word_in_text
finder.nbest(bigram_measures.pmi, 5)
#print(output)
print(Tokens)
print(bigram_fd.most_common(200), file=open("bigramsPOSFiltering.txt", "a"))
print(filtered_list, file=open("bigramsPOSFiltering.txt", "a"))
print(nltk.pos_tag(word_in_text), file=open("bigramsPOSFiltering.txt", "a"))
print(frequency_distribution.most_common(200))
print(finder.nbest(bigram_measures.likelihood_ratio, 10))
print(text_scored, file=open("bigramsPOSFiltering.txt", "a"))


#Trigrams
text = "Eşe karşı basit yaralama suçundan sanık ...'ın, 5237 sayılı Türk Ceza Kanunu'nun 86/2, 86/3-a, 62/1 ve 52/2. maddeleri uyarınca 3.000,00 Türk Lirası adlî para cezası ile cezalandırılmasına, 5271 sayılı Ceza Muhakemesi Kanunu'nun 231/5. maddesi gereğince hükmün açıklanmasının geri bırakılmasına dair Kırıkkale 4. Asliye Ceza Mahkemesinin 07.11.2019 tarihli ve 2018/535 Esas, 2019/948 Karar sayılı kararına karşı yapılan itirazın kabulü ile anılan kararın kaldırılmasına ilişkin mercii Kırıkkale 1. Ağır Ceza Mahkemesinin 05.02.2020 tarihli ve 2020/139 değişik iş sayılı kararına karşı Adalet Bakanlığının 17.11.2020 tarihli ve 2020/9153 sayılı yazısıyla kanun yararına bozma isteminde bulunulduğundan bu işe ait dava dosyası Yargıtay Cumhuriyet Başsavcılığının 15.12.2020 tarihli ve 2020/105078 sayılı tebliğnamesi ile Dairemize gönderilmekle incelendi.  Mezkur ihbarnamede;  5271 sayılı Kanun’un 231/8. maddesinde yer alan “Denetim süresi içinde, kişi hakkında kasıtlı bir suç nedeniyle bir daha hükmün açıklanmasının geri bırakılmasına karar verilemez.” şeklindeki düzenleme gereğince, denetim süresinde bir kez daha suç işlenmesi halinde ikinci suçtan dolayı kurulan hükmün tekrar açıklanmasının geri bırakılamayacağı nazara alındığında, sanık hakkında 06.06.2017 tarihinde işlemiş olduğu suçtan dolayı Kırıkkale 3. Asliye Ceza Mahkemesinin 10.05.2018 tarihli ve 2017/523 Esas, 2018/315 sayılı kararıyla verilen hükmün açıklanmasının geri bırakılmasına dair kararın 08.10.2018 tarihinde kesinleştiği ve denetim süresinin de bu tarih itibariyle başladığı, yargılama konusu suçun ise önceki kararın kesinleşmesinden önce 21.05.2018 tarihinde işlendiği cihetle, denetim süresi içerisinde işlenmiş bir suçtan bahsedilemeyeceği gözetilmeden, itirazın reddi yerine yazılı şekilde kabulüne karar verilmesinde isabet görülmediğinden bahisle, 5271 sayılı CMK'nin 309. maddesi gereğince anılan kararların bozulması lüzumunun ihbar olunduğu anlaşıldı.  Gereği görüşülüp düşünüldü:  5271 sayılı CMK’nin 231. maddesinde düzenlenen “hükmün açıklanmasının geri bırakılması” müessesesinin uygulanabilmesi için öncelikle,  - Sanık hakkında kurulan mahkûmiyet hükmünde, hükmolunan cezanın iki yıl veya daha az süreli hapis veya adli para cezasından ibaret olması,  - Suçun CMK’nın 231. maddesinin 14. fıkrasında yazılı suçlardan olmaması,  - Sanığın daha önce kasıtlı bir suçtan mahkûm olmamış bulunması,  - Sanığın hükmün açıklanmasının geri bırakılmasına itirazının bulunmaması,  Suçun işlenmesiyle mağdurun veya kamunun uğradığı zararın, aynen iade, suçtan önceki hale getirme veya tamamen giderilmesine ilişkin koşulların birlikte gerçekleşmesi gerekmektedir.  Ayrıca, bahsi geçen maddenin 8. fıkrasında; \"Hükmün açıklanmasının geri bırakılması kararının verilmesi halinde sanık, beş yıl süreyle denetim süresine tâbi tutulur. (Ek cümle: 18/06/2014-6545 S.K./72. md) Denetim süresi içinde, kişi hakkında kasıtlı bir suç nedeniyle bir daha hükmün açıklanmasının geri bırakılmasına karar verilemez.” hükmü yer almaktadır.  CMK’nin 231/8. maddesine ilişkin 6545 sayılı Kanun’un 72. maddesinin gerekçesinde de bu durum; “Maddeyle, Ceza Muhakemesi Kanunu'nun 231. maddesinin sekizinci fıkrasında değişiklik yapmak suretiyle, hükmün açıklanmasının geri bırakılmasına karar verilmesi halinde sanığın tabi tutulacağı denetim süresi içinde sanık hakkında bir daha hükmün açıklanmasının geri bırakılmasına karar verilemeyeceği düzenlenmektedir. Söz konusu maddenin uygulanmasında, hükmün açıklanmasının geri bırakılması kararı verilen sanıklar hakkında işledikleri diğer suçlardan dolayı da birçok kez hükmün açıklanmasının geri bırakılması kararı verildiği görülmektedir. Yapılması öngörülen değişiklikle, bu uygulamaya son verilmesi ve denetim süresi içinde sanık hakkında bir daha hükmün açıklanmasının geri bırakılmasına karar verilememesi amaçlanmaktadır. Kişinin işlediği ikinci suçun denetim süresi içinde işlenip işlenmediğinin önemi bulunmamaktadır. Daha önceden işlenen suçlar bakımından da bu yasak uygulanacaktır.” şeklinde ifade edilmiştir.  Buna göre 6545 sayılı Kanun'un yürürlüğe girdiği tarihten sonra işlenen suçlar için, hakkında daha önce hükmün açıklanmasının geri bırakılması kararı bulunan sanıklarla ilgili bir daha hükmün açıklanmasının geri bırakılması kararı verilemeyecektir.  İnceleme konusu somut olayda; mahkemece sanık ...’ın kasten basit yaralama suçundan adli para cezası ile cezalandırılmasına ve hükmün açıklanmasının geri bırakılmasına karar verilmiştir. Bu karara karşı yapılan itiraz merciince kabul edilerek sanık hakkındaki hükmün açıklanmasının geri bırakılmasına dair karar kaldırılmıştır.  Sanığın adli sicil kaydında yer alan kasten basit yaralama suçundan verilen Kırıkkale 3. Asliye Ceza Mahkemesinin 10.05.2018 tarihli ve 2017/523 Esas, 2018/315 Karar sayılı hükmün açıklanmasının geri bırakılması kararının 08.10.2018 tarihinde kesinleşmesi üzerine bu suç yönünden sanık hakkında denetim süresi başlamıştır. Böylece CMK’nin 231/8. maddesindeki; “Denetim süresi içinde, kişi hakkında kasıtlı bir suç nedeniyle bir daha hükmün açıklanmasının geri bırakılmasına karar verilemez” şeklindeki düzenleme gereğince inceleme konusu kasten basit yaralama suçu yönünden sanık hakkında hükmün açıklanmasının geri bırakılmasına karar verilemeyecektir. Sanığın inceleme konusu kasten basit yaralama suçunu adli sicil kaydındaki diğer kasten basit yaralama suçundan verilen hükmün açıklanmasının geri bırakılması kararının kesinleşme tarihinden önceki bir tarihte gerçekleştirmiş olmasının önemi bulunmamaktadır. Zira sanığın bu suçtan verilen hükmün açıklanmasının geri bırakılması kararının denetim süresi 08.10.2018 tarihinde başlamış ve incelenen kasten basit yaralama suçundan mahkemece 07.11.2019 tarihinde karar verilmiştir. 08.10.2018 tarihinden sonra sanık hakkında kasıtlı bir suçtan yeni bir hükmün açıklanmasının geri bırakılmasına karar verilmesi mümkün bulunmamaktadır.  Böylece, sanık hakkında mahkemece hükmün açıklanmasının geri bırakılmasına dair karara yönelik itirazın merciince kabul edilerek kaldırılmasına karar verilmesinde isabetsizlik görülmemiştir.  Açıklanan bu nedenlerle, Adalet Bakanlığının kanun yararına bozma isteyen yazısına dayanan tebliğnamede ileri sürülen düşünce yerinde görülmeyerek kanun yararına bozma talebinin REDDİNE, dosyanın mahalline gönderilmek üzere Yargıtay Cumhuriyet Başsavcılığına TEVDİİNE, 04.01.2021 gününde oybirliği ile karar verildi. (¤¤)"
wordlist = text.split()
trigram_fd = nltk.FreqDist(nltk.trigrams(wordlist))
trigram_measures = nltk.collocations.TrigramAssocMeasures()
trigram_fd.most_common()
words = [w.lower() for w in text]
finder = TrigramCollocationFinder.from_words(text)
finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10)
finder = TrigramCollocationFinder.from_words(text)
finder = TrigramCollocationFinder.from_words(filtered_list)
Tokens = nltk.word_tokenize(text)
output = tuple(nltk.trigrams(Tokens))
trigram_measures = nltk.collocations.TrigramAssocMeasures()
Tokens = nltk.word_tokenize(text)
finder = TrigramCollocationFinder.from_words(text)
text_finder = TrigramCollocationFinder.from_words(text)
text_scor = text_finder.score_ngrams(trigram_measures.raw_freq)
finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10)
output = tuple(nltk.trigrams(Tokens))
print(output)
print(trigram_fd.most_common(200), file=open("trigramsPMI.txt", "a"))
print(finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10))
print(text_scor)













