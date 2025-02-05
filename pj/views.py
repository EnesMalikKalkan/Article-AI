from django.shortcuts import render
from pj.forms import SignUpForm 
import pymongo, os, string, json, nltk, fasttext, re
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import fasttext.util
from transformers import AutoModel,AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

#fasttext.util.download_model('en', if_exists='ignore')
#ft = fasttext.load_model('cc.en.300.bin')
ft_model = None
dataset = load_dataset("memray/krapivin")
nltk.download('punkt')
path = "finished.json"
names = []
titles = []
abstracts = []
fulltexts = []
keywords = []
dizes = []
abstractVectors = []
abstractVectorsFasttext = []
kadi = ""
kgen = ""

if os.path.exists(path):
    print("Bu dosya sisteminizde bulunmaktadır .")
else:
    print("Bu dosya sisteminizde bulunmamaktadır .")
    with open(path, "w") as file:
        print("Dosya sistemde ilgili alan içerisinde oluşturuldu .")
# title-abstract-fulltext-keywords
if os.path.getsize(path) == 0:
    stopWords = set(stopwords.words('english'))
    noktalamaIsaretleri = set(string.punctuation)
    temizlenmisVeri = []
    stemmer = PorterStemmer()

    for datasetName in ['validation', 'test']:
        veri = dataset[datasetName]
        for oge in veri:
            temizlenmisOge = {}
            for sutun in oge.keys():
                if sutun in ['name', 'title', 'abstract', 'fulltext', 'keywords']:
                    if isinstance(oge[sutun], str):
                        kelimeler = word_tokenize(oge[sutun])
                        temizlenmisKelimeListesi = [stemmer.stem(kelime.lower()) for kelime in kelimeler if (kelime.lower() not in stopWords and kelime.lower() not in noktalamaIsaretleri)]
                        temizlenmisOge[sutun] = temizlenmisKelimeListesi
                    else:
                        temizlenmisOge[sutun] = oge[sutun]
            temizlenmisVeri.append(temizlenmisOge)

    with open(path, "w") as ciktiDosya:
        json.dump(temizlenmisVeri, ciktiDosya, indent=4)
else:
    with open(path, "r", encoding='utf-8') as okunacakDosya:
        okunanVeri = json.load(okunacakDosya)
        for oVeri in okunanVeri:
            names.append(oVeri["name"])
            titles.append(oVeri["title"])
            abstracts.append(oVeri["abstract"])
            fulltexts.append(oVeri["fulltext"])
            keywords.append(oVeri["keywords"])

    #flattenKeywords = [keyword for sublist in keywords for keyword in sublist]
    #keywordFrequency = Counter(flattenKeywords)
    #print(keywordFrequency.most_common(24))
    #for abstract in abstracts:
       # dize = ' '.join(abstract)  
       #dizes.append(dize)  
       #abstract.clear() 
    
    dizes = [' '.join(inner_list) for inner_list in abstracts]
    dizesi = ' '.join(map(str,dizes))
    
    file_path = "ozetler.txt"

    with open (file_path, "w" , encoding = "utf-8") as data_file:
        data_file.write(dizesi)
    
    # FastText ile özetlerden vektörler oluştur
    ft_model = fasttext.train_unsupervised(input= "ozetler.txt", epoch = 3)
    
    for i in range(460):
        abstractVectorsFasttext.append(ft_model.get_sentence_vector(dizes[i]))
    
    # ---------------> Scibert <---------------#

    scibertName = "allenai/scibert_scivocab_uncased"
    model = AutoModel.from_pretrained(scibertName)
    tokenizer = AutoTokenizer.from_pretrained(scibertName)

    def embed_articles(dizes):                               # Makaleleri gömmek için fonksiyona gönderdik
        embedded_articles = []

    #Her makale metnini SciBERT modeline besleyerek gömme vektörlerini elde et
        for i in range(460):
            # Makale metnini tokenleştir
            makaleler = dizes[i]
            inputs = tokenizer(makaleler, max_length=512 , return_tensors="pt", truncation=True)

    #SciBERT modeline gönder ve gömülme vektörlerini al
            with torch.no_grad():
                outputs = model(**inputs)

    #Son gizli katmanın çıktısını al (gömme vektörleri)
            embeddings = outputs.last_hidden_state

    #Gömülme vektörlerinin ortalamasını alarak makaleyi temsil et
            article_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    #Temsil edilen makaleyi gömme vektörleri listesine ekle
            embedded_articles.append(article_embedding)

        return embedded_articles


    abstractVectors = embed_articles(dizes)

    #print(abstractVectors[1])


def home(request):
    return render(request, 'main_page.html')

def kayit(request):
    return render(request, 'register.html')

def connectDB():
    myClient = pymongo.MongoClient("mongodb://localhost:27017/")
    myDB = myClient["academic"]
    return myDB

def addUser(request):
    global kadi, kgen
    dB = connectDB()
    userCollection = dB["users"]
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        userName = request.POST["signup_uname"]
        userMail = request.POST["signup_mail"]
        userBdate = request.POST["signup_birthdate"]
        userGender = request.POST["signup_gender"]
        userPasswd = request.POST["signup_pass"]
        userInterests = request.POST.getlist('interest')

        userDict = {"username": userName, "mail": userMail, "birthdate": userBdate, "gender": userGender, "password": userPasswd, "interests": userInterests}
        dbDataDict = {"username": userName, "mail": userMail}
        
        uControl = controlInformations(dbDataDict)

        if uControl:
            error_message = "Girilen kullanıcı adı ve e-posta sisteme kayıtlı"
            return render(request, 'register.html', {'error_message': error_message})
        else:
            add = userCollection.insert_one(userDict)
            kadi = userName
            kgen = userGender
            findRecommendations(request, userInterests, userName, userGender)

    return render(request, 'register.html')

def updateUser(request):
    global kadi
    dB = connectDB()
    userCollection = dB["users"]
    if request.method == 'POST':
        userName = request.POST["username"]
        userMail = request.POST["email"]
        userBdate = request.POST["birthdate"]
        userGender = request.POST["gender"]
        if userGender == "Erkek":
            userGender = "Male"
        else:
            userGender = "Female"
        userPasswd = request.POST["password"]

        usr = kadi

        userDict = {"username": userName, "mail": userMail, "birthdate": userBdate, "gender": userGender, "password": userPasswd}
        
        updt = userCollection.update_one({'username': usr}, {'$set': userDict})

        if updt.modified_count > 0:
            getUsr = userCollection.find_one({"username": userName, "mail": userMail})
            if getUsr:
                interests = getUsr.get("interests")
                kadi = userName
                return findRecommendations(request, interests, userName, userGender)
        else:
            return editProfile(request)

def deleteUser(request):
    return True

def controlInformations(dbDataDict):
    dB = connectDB()
    userCollection = dB["users"]
    control = userCollection.find_one(dbDataDict)
    return control

def findRecommendations(request, userInterests, userName, gender):
    global abstractVectors, abstractVectorsFasttext, ft_model, titles, abstracts, model, tokenizer
    
    # FastText kullanıcı vektör ortalaması kısmı
    interestFasttext = []
    for interest in userInterests:
        vector = ft_model.get_word_vector(interest)
        interestFasttext.append(vector)
    interestFasttextAvg = sum(interestFasttext) / len(interestFasttext)
    interestFasttextAvg = interestFasttextAvg[np.newaxis, :]
    
    # SciBERT kullanıcı vektör ortalaması kısmı
    def encode_text_list(text_list):
        encoded = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
        return encoded
    
    encoded_veri = encode_text_list(userInterests)
    
    with torch.no_grad():
        outputs = model(**encoded_veri)
    
    # Only use [CLS] token (first token) for each input interest
    cls_vectors = outputs.last_hidden_state[:, 0, :].numpy()
    
    interestScibertAvg = np.mean(cls_vectors, axis=0)
    interestScibertAvg = interestScibertAvg[np.newaxis, :]
    
    similaritiesScibert = cosine_similarity(abstractVectors, interestScibertAvg)
    similaritiesFasttext = cosine_similarity(abstractVectorsFasttext, interestFasttextAvg)
    
    top5ScibertIndices = similaritiesScibert.flatten().argsort()[-5:][::-1]
    top5FasttextIndices = similaritiesFasttext.flatten().argsort()[-5:][::-1]

    dizeTitle = [' '.join(inner_list) for inner_list in titles]
    dizeAbstracts = [' '.join(inner_list) for inner_list in abstracts]
    
    dataScibert = [{'title': dizeTitle[idx].upper(), 'similarity': similaritiesScibert.flatten()[idx], 'abstract': dizeAbstracts[idx]} for idx in top5ScibertIndices]
    dataFasttext = [{'title': dizeTitle[idx].upper(), 'similarity': similaritiesFasttext.flatten()[idx], 'abstract': dizeAbstracts[idx]} for idx in top5FasttextIndices]

    return render(request, 'innerlog.html', {
        'username': userName,
        'gender': gender,
        'articlesScibert': dataScibert,
        'articlesFasttext': dataFasttext
    })

def validateUser(request):
    global kadi, kgen
    dB = connectDB()
    userCollection = dB["users"]
    
    if request.method == 'POST':
        uname = request.POST["login_uname"]
        passwd = request.POST["login_pass"]
        
        findDict = {'username': uname, 'password': passwd}
        
        result = userCollection.find_one(findDict)
        
        if result:
            kadi = uname
            interests = result.get("interests")
            gender = result.get("gender")
            kgen = gender
            return findRecommendations(request, interests, uname, gender)
        else:
            return render(request, 'register.html')
        
def editProfile(request):
    global kadi
    dB = connectDB()
    userCollection = dB["users"]
    result = userCollection.find_one({'username': kadi})
    mail = result.get("mail")
    bdate = result.get("birthdate")
    gender = result.get("gender")
    if gender == "Male":
        gender = "Erkek"
    else:
        gender = "Kadın"
    return render(request, 'edit.html', {'user': kadi, 'username': kadi, 'genderi': gender, 'gender': gender, 'mail': mail, 'birthdate': bdate})

def listResults(request):
    global dataset, titles, abstracts, keywords, names, kgen
    if request.method == 'POST':
        keyword = request.POST.get("query")
        validationData = dataset['validation']
        testData = dataset['test']
        
        def filterFunction(example):
            return keyword.lower() in example['keywords'].lower()
        
        filteredValidation = list(filter(filterFunction, validationData))
        filteredTest = list(filter(filterFunction, testData))
        
        combinedFiltered = filteredValidation + filteredTest
        top10 = [item['name'] for item in combinedFiltered[:10]]

        dizeNames = [' '.join(inner_list) for inner_list in names]
        
        indices = []
        for name in top10:
            index_in_dizeNames = dizeNames.index(name)
            indices.append(index_in_dizeNames)

        dizeTitle = [' '.join(inner_list) for inner_list in titles]
        dizeAbstracts = [' '.join(inner_list) for inner_list in abstracts]
        dizeKeywords = [' '.join(inner_list) for inner_list in keywords]

        datas = [{'title': dizeTitle[idx].upper(), 'abstract': dizeAbstracts[idx], 'keywords': dizeKeywords[idx]} for idx in indices]
        
        return render(request, 'list.html', {'searchedWord': keyword, 'gender': kgen, 'articles': datas})

def showFullPage(request):
    global titles, names, fulltexts, kadi, kgen
    if request.method == 'POST':
        choosenTitle = request.POST.get("title")
        choosenAbstract = request.POST.get("abstract")
        choosenKeywords = request.POST.get("keywords")
        searchedWord = request.POST.get("searchedWord")

        choosenTitle = choosenTitle.lower()

        dizeTitles = [' '.join(inner_list) for inner_list in titles]
        dizeFulltext = [' '.join(inner_list) for inner_list in fulltexts]

        indice = dizeTitles.index(choosenTitle)
        searchedWordLower = searchedWord.lower()
        
        dB = connectDB()
        userCollection = dB["users"]

        usr = kadi

        user = userCollection.find_one({'username': usr})
        user = userCollection.find_one({'username': usr})
        if user and 'interests' in user:
            interests_lower = [interest.lower() for interest in user['interests']]
            if searchedWordLower in interests_lower:
                alreadyExists = True
            else:
                userCollection.update_one({'username': usr}, {'$push': {'interests': searchedWord}})
                alreadyExists = False
        else:
            userCollection.update_one({'username': usr}, {'$push': {'interests': searchedWord}})
            alreadyExists = False

        return render(request, 'result.html', {
            'gender': kgen,
            'searchedWord': searchedWord,
            'title': choosenTitle.upper(),
            'abstract': choosenAbstract,
            'keywords': choosenKeywords,
            'fulltext': dizeFulltext[indice],
            'name': names[indice]
        })
    
def showFullPageRecommendation(request):
    global keywords, fulltexts, names, titles
    if request.method == 'POST':
        recommendationTitle = request.POST.get("title")
        recommendationAbstract = request.POST.get("abstract")

        recommendationTitle = re.sub(r' - .*$', '', recommendationTitle).strip()
        recommendationTitle = recommendationTitle.lower()

        dizeTitles = [' '.join(inner_list) for inner_list in titles]
        dizeFulltext = [' '.join(inner_list) for inner_list in fulltexts]
        dizeKeywords = [' '.join(inner_list) for inner_list in keywords]

        indice = dizeTitles.index(recommendationTitle)

        #bir üstteki listeleme kısmının index yerine bak orlara yanlış olabilir

        return render(request, 'result.html', {
            'gender': kgen,
            'searchedWord': "Recommendation",
            'title': recommendationTitle.upper(),
            'abstract': recommendationAbstract,
            'keywords': dizeKeywords[indice],
            'fulltext': dizeFulltext[indice],
            'name': names[indice]
        })