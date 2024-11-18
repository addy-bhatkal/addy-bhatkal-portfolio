from django.shortcuts import render
import pickle 
import sklearn
import pandas as pd
import warnings
from django.contrib import messages
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

from django.shortcuts import redirect

def tableau_redirect(request):
    return redirect("https://public.tableau.com/app/profile/addy.bhatkal/favorites")


def home(request):
    routes = [
        {'endpoint': 'Movie Recommender', 'url': "movierecommender/", 'image':'images/movierecommender.png'},
        {'endpoint': 'Frequently Bought Items', 'url': "arm/", 'image':'images/arm.png'},
        {'endpoint': 'Diabetes Predictor (Decision Tree Classifier)', 'url': 'diabetes/','image':'images/diabetes2.png'},
        {'endpoint': 'News Category Classifier (Count Vectorizer + Naive Bayes)', 'url': 'nclassifier/', 'image':'images/news3.png'},
        {'endpoint': 'Song Recommender (Term Frequency-Inverse Document Frequency + Cosine Similarity)', 'url': 'recommend_songs/', 'image':'images/song2.png'},
        {'endpoint': 'Customer Marketing Campaign Buy-in (Light Gradient Boost)', 'url': 'customerbuyin/', 'image':'images/cby2.png'},
        {'endpoint': 'Bank Credit Card Customer Churn (Logistic Regression)', 'url': "bankchurn/", 'image':'images/bank.png'},
        {'endpoint': 'Data Visualizations: Tableau', 'url': "DataViz/", 'image': 'images/dataviz.png'}, 
        {'endpoint': 'HousingPrices', 'url': "coming-soon/", 'image': 'images/uc.png'}
        
    ]
    return render(request, 'home.html', {'routes': routes})



def comingsoon1(request):
    return render(request, 'comingsoon1.html')

with open('diabetes_predictor.sav', 'rb') as f2:
    load_model = pickle.load(f2)

def infer_diabetes(ip_data): #ML part of code
    prob = load_model.predict_proba(ip_data)[0][1]
    return load_model.predict_proba(ip_data)[0][1] >0.3, f"The probability with which you will suffer from diabetes is: {prob*100 :.2f} %"

def diabetes(request): #HTML part of code
    context = dict()
    
    """f request.method == 'GET':
        print(request.GET.get("manoj"))  # Initialize the context dictionary
"""
    if request.method == 'POST':
        try:
            pregnancies = int(request.POST['Pregnancies'])
            glucose = float(request.POST['Glucose'])
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


df = pd.read_csv('spotify_millsongdata2.csv')

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


def song_list(request):
    songs_list = df['song_artist'].tolist()  # Assuming the column name is 'name'
    paginator = Paginator(songs_list, 200) 
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Calculate the page range (divide into 10-page chunks)
    current_page = page_obj.number
    chunk_size = 10  # Set the chunk size (can be 10, 20, etc.)
    chunk_start = ((current_page - 1) // chunk_size) * chunk_size + 1
    chunk_end = min(chunk_start + chunk_size - 1, paginator.num_pages)
    page_range = range(chunk_start, chunk_end + 1)

    return render(request, 'song_list.html', {
        'page_obj': page_obj,
        'page_range': page_range,
    })

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

df2 = pd.read_csv('ARM - Bakery_mapped2.csv')


def arm_ML(prod):
    df2['Items'] = df2['Items'].str.strip()
    basket = df2.groupby(['TransactionNo','Items'])['DateTime'].sum().unstack().reset_index().fillna(0).set_index('TransactionNo')
    def encode(y):
        if y== 0:
            return 0
        if y!= 0:
            return 1
    basket = basket.map(encode)
    freq = apriori(basket, min_support=0.005, use_colnames=True)
    rules = association_rules(freq, freq['itemsets'], metric='lift', min_threshold=1)
    rules2 = rules.sort_values(by='lift', ascending = False)
    df_rules = rules2.reset_index()
    idx = [i for i in range(df_rules.shape[0])]
    df3 = df_rules[['antecedents', 'consequents']].iloc[idx,:].reset_index().drop('index', axis = 1)
    df3['antecedents'] = df3['antecedents'].apply(lambda x: ', '.join(list(x)))
    df3['consequents'] = df3['consequents'].apply(lambda x: ', '.join(list(x)))
    df3.head()
    return list(df3[df3['antecedents']==prod]['consequents'])

def arm(request):
    product = None
    similar_products = None

    if request.method == 'POST':
        product = request.POST.get('product', '')
        similar_products = arm_ML(product)
        print(similar_products)

    return render(request, 'arm.html', {'product': product, 'similar_products': similar_products})



movie_df = pd.read_csv('moviesrecom.csv')
warnings.filterwarnings("ignore", category=RuntimeWarning)

def movierecom_ML(moviename):
    #if moviename not in movie_df.columns:     # Check if the movie exists in the DataFrame
        #return pd.DataFrame()  # Return an empty DataFrame if the movie does not exist
    movie_res = movie_df.corrwith(movie_df[moviename]).sort_values(ascending=False)
    movie_res2 = pd.DataFrame(movie_res, columns=['corr']).reset_index().rename(columns={'index': 'movies'})
    #movie_res2 = movie_res2.reset_index().rename(columns={'index': 'movies'})
    return movie_res2[['movies', 'corr']].head(50)

from django.core.paginator import Paginator

def movierecom(request): 
    movie = None
    similar_movie = None

    try: 
        if request.method == "POST":
            movie = request.POST.get('movie', '')
            movie = movie.strip()
            if movie:
                similar_movie_df = movierecom_ML(movie) 
                similar_movie = similar_movie_df[['movies']].values.tolist() # Convert DataFrame to a list of tuples (movie name, correlation)
            else:
                similar_movie = []

    except ValueError:
            similar_movie = []

    return render(request, 'movierecommender.html', {'movie': movie, 'similar_movie': similar_movie if similar_movie is not None else []})


def movie_list(request):
    movies = movie_df.columns.tolist()  # Assuming the column name is 'name'
    paginator = Paginator(movies, 50)  # Show 50 movies per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Calculate the page range (1-10 initially)
    current_page = page_obj.number
    page_range_start = max(1, ((current_page - 1) // 10) * 10 + 1)
    page_range_end = min(page_range_start + 9, paginator.num_pages)
    page_range = range(page_range_start, page_range_end + 1)

    return render(request, 'movie_list.html', {
        'page_obj': page_obj,
        'page_range': page_range,
        'r' : range(1,11)
    })
