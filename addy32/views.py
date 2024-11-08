from django.shortcuts import render
import pickle 
import sklearn
import pandas as pd

from django.contrib import messages
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD



def home(request):
    routes = [
        {'endpoint': 'diabetes', 'url': 'diabetes/'},
        {'endpoint': 'nclassifier', 'url': 'nclassifier/'},
        {'endpoint': 'recommend_songs', 'url': 'recommend_songs/'},
        {'endpoint': 'customerbuyin', 'url': 'customerbuyin/'},
        {'endpoint': 'bankchurn', 'url': "bankchurn/"}

        
    ]
    return render(request, 'home.html', {'routes': routes})

with open('diabetes_predictor.sav', 'rb') as f2:
    load_model = pickle.load(f2)

def infer_diabetes(ip_data): #ML part of code
    prob = load_model.predict_proba(ip_data)[0][1]
    return load_model.predict(ip_data)[0] == 1, f"The probability with which you will suffer from diabetes is: {prob*100 :.2f} %"

def diabetes(request): #HTML part of code
    context = dict()
    
    """f request.method == 'GET':
        print(request.GET.get("manoj"))  # Initialize the context dictionary
"""
    if request.method == 'POST':
        try:
            pregnancies = int(request.POST['Pregnancies'])
            glucose = int(request.POST['Glucose'])
            blood_pressure = int(request.POST['BloodPressure'])
            skin_thickness = int(request.POST['SkinThickness'])
            insulin = int(request.POST['Insulin'])
            bmi = float(request.POST['BMI'])
            diabetes_pedigree = float(request.POST['DiabetesPedigreeFunction'])
            age = int(request.POST['Age'])

            data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]

            result,probability = infer_diabetes(data)  # Assuming infer_diabetes is imported and works properly

            if result:
                context['result'] = f"Diabetes detected! {probability}"
                context['result_class'] = "danger"
            else:
                context['result'] = f"No diabetes detected. {probability}"
                context['result_class'] = "success"

            # Pass the input values back to the template
            context.update({
                'pregnancies': pregnancies,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'bmi': bmi,
                'diabetes_pedigree': diabetes_pedigree,
                'age': age
            })

        except ValueError:
            context['result'] = "Invalid input! Please enter valid numbers."
            context['result_class'] = "danger" #the class danger means the text will pop up in red

    return render(request, 'diabetes.html', context)


with open('cv1.pkl', 'rb') as cv1_file, open('nb_model.pkl', 'rb') as model1_file:
    cv1 = pickle.load(cv1_file)
    model1 = pickle.load(model1_file)

def nclassifier_ML(user_input):
    user_input_array = cv1.transform([user_input]).toarray()
    classification =  model1.predict(user_input_array)
    return classification[0]


def nclassifier(request):
    if request.method == 'POST':
        article = request.POST.get('article', '')
        result = nclassifier_ML(article)
        return render(request, 'nclassifier.html', {'article': article, 'result':result})
    return render(request, 'nclassifier.html', {'article': ''}) 


df = pd.read_csv('spotify_millsongdata.csv')

tf = TfidfVectorizer(ngram_range=(1,2), max_df=0.5, min_df=5, stop_words='english')
vec = tf.fit_transform(df['text'])

def reco(song):
    lsa = TruncatedSVD(n_components=100)  # Choose number of dimensions to reduce to
    lsa_matrix = lsa.fit_transform(vec)
    val1 = tf.transform([song])
    val2 = lsa.transform(val1)
    sim = cosine_similarity(lsa_matrix, val2)
    distances = sorted(list(enumerate(sim)), reverse= True, key = lambda x : x[1])
    listy = []
    for i in distances[:15]:
        listy.append([df.iloc[i[0]][0], df.iloc[i[0]][1]])
        #listy2 = pd.DataFrame(listy)
    return listy

def song_reco(request): 
    song = None
    similar_songs = None

    if request.method == "POST":
        song = request.POST.get('song', '')
        similar_songs = reco(song)
        

    return render(request, 'recommend_songs.html', {'similar_songs':similar_songs, 'song':song })

with open('customerbuyin.sav', 'rb') as f2:
    cby_loaded_model = pickle.load(f2)

def cby_ML(data):
    prob = float(cby_loaded_model.predict_proba(data)[0][1])
    return int(cby_loaded_model.predict(data)[0]) ==1, f'The probability that this customer will buy into your current Ad campaign is {prob*100 :.2f} %'

def cby(request):
    context = {}

    if request.method == 'POST':
        try:
            education = int(request.POST['Education'])
            maritalstatus = int(request.POST['Marital_Status'])
            income = int(request.POST['Income'])
            kidhome = int(request.POST['Kidhome'])
            teenhome = int(request.POST['Teenhome'])
            mntfruits = int(request.POST['MntFruits'])
            newwebpurchases = int(request.POST['NumWebPurchases'])
            numcatalogpurchases = int(request.POST['NumCatalogPurchases'])
            numstorepurchases = int(request.POST['NumStorePurchases'])
            numwebvisitsmonth = int(request.POST['NumWebVisitsMonth'])
            complain = int(request.POST['Complain'])

            cby_data = [[education, maritalstatus, income, kidhome, teenhome, mntfruits, newwebpurchases, numcatalogpurchases, numstorepurchases, numwebvisitsmonth, complain]]

            result, prob = cby_ML(cby_data)

            if result: 
                context['result'] = f' YES. {prob}'
                context['result_class'] = 'success'
            else: 
                context['result'] = f'No {prob}'
                context['result_class'] = 'success'

            context.update({
                'education': education, 
                'maritalstatus': maritalstatus, 
                'income': income, 
                'kidhome': kidhome, 
                'teenhome': teenhome, 
                'mntfruits': mntfruits, 
                'newwebpurchases': newwebpurchases, 
                'numcatalogpurchases': numcatalogpurchases, 
                'numstorepurchases': numstorepurchases, 
                'numwebvisitsmonth': numwebvisitsmonth, 
                'complain': complain
            }) #If 0 then python sees it as nothing 

        except ValueError:
            context['result'] = "Invalid Input"
            context['result_class'] = 'danger'


    return render(request, 'customerbuyin.html', context)

with open('bankchurn.sav', 'rb') as f2:
    loaded_bankchurn = pickle.load(f2)

def bankchurn_ML(bc_data): #for ML code
    probability = loaded_bankchurn.predict_proba(bc_data)[0][1]
    return loaded_bankchurn.predict(bc_data)[0] == 1, f'The probability with which this customer will churn is {probability*100 :.2f} %'

def bankchurn(request):
    context = {}

    if request.method == 'POST':
        try:
            creditscore = int(request.POST['CreditScore']) 
            gender = int(request.POST['Gender'])
            age = int(request.POST['Age'])
            tenure = int(request.POST['Tenure'])
            balance = int(request.POST['Balance'])
            numofproducts = int(request.POST['NumOfProducts'])
            hasacard = int(request.POST['HasCrCard'])
            isaactivemember = int(request.POST['IsActiveMember']) 
            estimatedsalary = int(request.POST['EstimatedSalary'])

            bc_data = [[creditscore, gender, age, tenure, balance, numofproducts, hasacard, isaactivemember, estimatedsalary]]

            result, prob_HTML = bankchurn_ML(bc_data)

            if result: 
                context['result'] = f'CHURN DETECTED! {prob_HTML}'
                context['result_class'] = 'danger'
            else:
                context['result'] = f"'CHURN NOT DETECTED! {prob_HTML} "
                context['result_class'] = 'success'

            context.update({
                'creditscore': creditscore, 
                'gender':gender,
                'age':age,
                'tenure': tenure,
                'balance':balance,
                'numofproducts': numofproducts,
                'hasacard':hasacard,
                'isaactivemember': isaactivemember,
                'estimatedsalary': estimatedsalary
                })

        except ValueError:
            if result: 
                context['result'] = 'Invalid input. Please correct the incorrect values'
                context['result_class'] = 'danger'


    return render(request, 'bankchurn.html', context)


