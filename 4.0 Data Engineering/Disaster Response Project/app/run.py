import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords

app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes the input text by performing the following steps:
    1. Converts text to lowercase.
    2. Removes non-alphanumeric characters.
    3. Tokenizes the text into words.
    4. Removes English stopwords.
    5. Lemmatizes the tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of processed tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stopwords_to_remove = set(stopwords.words('english'))
    clean_tokens = [lemmatizer.lemmatize(token).strip() for token in tokens if token not in stopwords_to_remove]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
for column in df.columns[4:]:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df['genre'].value_counts()
    genre_names = list(genre_counts.index)

    related_counts = df['related'].value_counts()
    related_names = ['Related', 'Not Related']

    categories = df.iloc[:, 4:].sum().sort_values(ascending=False).head(15)
    category_names = list(categories.index)
    category_counts = list(categories.values)

    direct_report_counts = df['direct_report'].value_counts()
    direct_report_names = ['Direct Report', 'Not Direct Report']
        
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hole=.3
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts,
                    marker=dict(color=['#1f77b4', '#ff7f0e']),
                    text=related_counts,
                    textposition='auto'
                )
            ],
            'layout': {
                'title': 'Distribution of Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker=dict(color='rgba(50, 171, 96, 0.6)'),
                    text=category_counts,
                    textposition='auto'
                )
            ],
            'layout': {
                'title': 'Top 10 Categories in Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_report_names,
                    y=direct_report_counts,
                    marker=dict(color=['#d62728', '#2ca02c']),
                    text=direct_report_counts,
                    textposition='auto'
                )
            ],
            'layout': {
                'title': 'Distribution of Direct Reports',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Direct Report"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    contact_info = {
        'related': 'Banana Bandits',
        'request': 'Wacky Wish Granters',
        'offer': 'Silly Supply Squad',
        'aid_related': 'Jolly Aid Jugglers',
        'medical_help': 'Quirky Quacks',
        'medical_products': 'Nutty Nurse Network',
        'search_and_rescue': 'Goofy Lifesavers',
        'security': 'Clown Patrol',
        'military': 'Giggly Guards',
        'child_alone': 'Lonely Lollipop League',
        'water': 'H2O Hoarders',
        'food': 'Munchie Masters',
        'shelter': 'Snug Bug Society',
        'clothing': 'Fashion Fools',
        'money': 'Cash Clowns',
        'missing_people': 'Hide and Seek Heroes',
        'refugees': 'Wandering Wackos',
        'death': 'Grim Grinners',
        'other_aid': 'Random Rascals',
        'infrastructure_related': 'Loony Builders',
        'transport': 'Crazy Cab Company',
        'buildings': 'Nutty Constructors',
        'electricity': 'Zany Zap Squad',
        'tools': 'Tool Time Troupe',
        'hospitals': 'Hospital Hijinks',
        'shops': 'Silly Shopkeepers',
        'aid_centers': 'Aid Antics',
        'other_infrastructure': 'Infrastructure Imps',
        'weather_related': 'Weather Whackos',
        'floods': 'Flood Fools',
        'storm': 'Storm Stooges',
        'fire': 'Fire Funnies',
        'earthquake': 'Quake Quacks',
        'cold': 'Chilly Chums',
        'other_weather': 'Weather Weirdos',
        'direct_report': 'Direct Report Dodos'
    }
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        contact_info=contact_info
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()