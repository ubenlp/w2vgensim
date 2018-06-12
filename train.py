# -*- coding: utf-8 -*-

import gensim
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.chdir('your directory path')

#Derlemi tutan değişken. 
data_file='wikitr.txt'

#Text dosyasını okuyan method. Ayrıca her bir cümle için gensim kütüphanesi 
#içerisinde bulunan simple_preprocess methodu ile cümle içerisindeki kelimeleri
#liste yapısında tutar.
def read_input(input_file):
    
    #Text dosyası okunurken bilgi veren log komutu.
    logging.info("Dosya okunuyor {0}...".format(input_file))
    
    #With open methodu ile text dosyasını açıyoruz.
    with open(input_file, 'rb') as file:
        
        #file değişkeninde tuttuğumuz text dosyamız enumerate methodu ile cümle
        #sayısını yazdırıyoruz.
        for i, line in enumerate(file): 

            if (i%10000==0):
                logging.info ("{0} satır okundu".format (i))
                
            #yield methodu ile pre process edilen cümleler saklanır.
            yield gensim.utils.simple_preprocess (line)

#Cümleleri liste yapısında tutuyoruz. Yapı şöyle olur:
            # 1 - Cümle 1
            # 2 - Cümle 2
            # 3 - Cümle 3
#Şeklinde gider. Böylelikle gensim'e list of list yapısında bir veri yapısı
#göndermiş oluruz.
sentences = list(read_input(data_file))
logging.info("Text dosyasını okuma tamamladı.")

#Gensim'in modeli oluşturması için  vocabulary oluşturmalıyız. Bunun aşağıdaki
#method ile yaparız.
"""
değişkenlerin anlamı:
    sentences: text dosyamız
    
    size: dense kelime vektörü boyutu
    
    window: merkezdeki kelime için pencere boyutu. Hedef kelime ile komşu 
    kelimesi arasındaki maksimum mesafe. Komşunuzun pozisyonu, sol veya 
    sağdaki maksimum pencere genişliğinden daha büyükse, bazı komşular 
    hedef kelime ile ilişkili olarak kabul edilmez. Teorik olarak, daha 
    küçük bir pencere size daha alakalı terimler vermelidir. Yine, verileriniz
    seyrek değilse, aşırı dar veya aşırı geniş olmadığı sürece pencere boyutu
    çok fazla önem taşımamaktadır. 
    
    min_count: dahil edilecek kelimelerin frekans treshold değeri. 
    workers: çalıştırılması istenilen thread yapıları.
"""
model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=2, workers=10)

#Modelimizi eğitmek için çağırılan method. 
model.train(sentences,total_examples=len(sentences),epochs=10)
