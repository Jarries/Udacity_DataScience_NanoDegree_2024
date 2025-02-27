import json
import pickle

import numpy as np
import pandas as pd
import plotly
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from plotly.graph_objs import Bar, Pie
from wtforms.fields import SelectField

app = Flask(__name__)


def load_model(model_filepath):
    """Load trained machine learning model."""
    with open(model_filepath, 'rb') as pickle_model:
        ml_model = pickle.load(pickle_model)
    pickle_model.close()
    return ml_model


# Load model
model = load_model('../models/random_forest_classifier.pkl')

# Load data
df = pd.read_csv('../data/data_for_analysis.csv')
portfolio = pd.read_csv('../data/engineered_portfolio.csv')
profile = pd.read_csv('../data/engineered_profile.csv')
app.config['SECRET_KEY'] = 'secret'

# need to check this out!
def get_offer_user_attributes(portfolio_after_engineer, profile_after_engineered, offer_id, user_id):
    """
    Get the offer and user attributes from DataFrames.

    This function retrieves the attributes of a specified offer and user from the engineered portfolio
    and profile DataFrames. It ensures that the provided offer ID and user ID exist in the respective
    DataFrames and then combines their attributes into a single array.

    Args:
        portfolio_after_engineer (pd.DataFrame): The engineered portfolio DataFrame.
        profile_after_engineered (pd.DataFrame): The engineered profile DataFrame.
        offer_id (str): The ID of the offer to retrieve attributes for.
        user_id (str): The ID of the user to retrieve attributes for.

    Returns:
        np.ndarray: An array containing the combined attributes of the specified offer and user.
    """
    assert offer_id in list(portfolio_after_engineer['offer_id']), f"Offer ID {offer_id} not found in portfolio DataFrame."
    assert user_id in list(profile_after_engineered['id']), f"User ID {user_id} not found in profile DataFrame."

    offer_values_dict = portfolio_after_engineer[portfolio_after_engineer['offer_id'] == offer_id].drop('offer_id',
                                                                                                axis=1).to_dict(
        'records')[0]
    user_values_dict = profile_after_engineered[profile_after_engineered['id'] == user_id].drop('id', axis=1).to_dict('records')[0]

    return offer_values_dict, user_values_dict

def predict(model, offer_values_dict, user_values_dict):
    """
    Make a prediction using a machine learning model.

    This function takes a trained machine learning model, engineered DataFrames for portfolio and profile,
    and specific offer and user IDs. It retrieves the corresponding attributes, combines them, and returns
    the model's prediction.

    Args:
        model (object): The trained machine learning model.
        portfolio_after_engineering (pd.DataFrame): The engineered DataFrame containing offer data.
        profile_after_engineering (pd.DataFrame): The engineered DataFrame containing user profile data.
        offer_id (str): The ID of the offer to retrieve attributes for.
        user_id (str): The ID of the user to retrieve attributes for.

    Returns:
        The prediction result from the model.
    """
       
    # Combine the offer and user attributes into a single array
    combined_values = list(offer_values_dict.values()) + list(user_values_dict.values())
    
    # Convert the combined values to a numpy array
    combined_values_array = np.array(combined_values).reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(combined_values_array)
    
    return prediction


def age_interval(age):
    """
    Categorize age into an interval.

    This function takes an age value and categorizes it into a predefined interval.

    Parameters:
    age (int): The age value to be categorized.

    Returns:
    str: The interval in which the age falls.

    Raises:
    ValueError: If the age value is not within the expected range.

    Example:
    >>> age_interval(25)
    '[20, 30)'

    The intervals are defined as follows:
    - '[0, 10)': Ages from 0 to 9
    - '[10, 20)': Ages from 10 to 19
    - '[20, 30)': Ages from 20 to 29
    - '[30, 40)': Ages from 30 to 39
    - '[40, 50)': Ages from 40 to 49
    - '[50, 60)': Ages from 50 to 59
    - '[60, 70)': Ages from 60 to 69
    - '[70+, ]': Ages 70 and above
    """
    if age < 10:
        interval = '[0, 10)'
    elif age >= 10 and age < 20:
        interval = '[10, 20)'
    elif age >= 20 and age < 30:
        interval = '[20, 30)'
    elif age >= 30 and age < 40:
        interval = '[30, 40)'
    elif age >= 40 and age < 50:
        interval = '[40, 50)'
    elif age >= 50 and age < 60:
        interval = '[50, 60)'
    elif age >= 60 and age < 70:
        interval = '[60, 70)'
    elif age >= 70:
        interval = '[70+, ]'
    else:
        raise ValueError(f"Unknown value for age: {age}")
    return interval


def income_interval(income):
    """
    Categorize income into an interval.

    Parameters:
    income (float): The income to be categorized.

    Returns:
    str: The interval in which the income falls.

    Raises:
    ValueError: If the income is not within the expected range.

    This function takes an income value as input, converts it to thousands, and returns the corresponding income interval as a string.
    """
    income = income / 1000
    if 30 <= income < 40:
        interval = '[30, 40)'
    elif 40 <= income < 50:
        interval = '[40, 50)'
    elif 50 <= income < 60:
        interval = '[50, 60)'
    elif 60 <= income < 70:
        interval = '[60, 70)'
    elif 70 <= income < 80:
        interval = '[70, 80)'
    elif 80 <= income < 90:
        interval = '[80, 90)'
    elif 90 <= income < 100:
        interval = '[90, 100)'
    elif income >= 100:
        interval = '[100+, ]'
    else:
        raise ValueError(f"Unknown value for income: {income}")
    return interval




# Index webpage displays cool visuals and receives user input text for model.
@app.route('/')
@app.route('/index')
def index():
    labels = ['Successful', 'Not-successful']
    success = np.mean(df['successful_offer'])
    values = [success, 1 - success]

    offer_types = df.groupby('offer_type')['successful_offer'].mean()
    offer_types_df = pd.DataFrame({'offer_type': offer_types.index, '% of success': offer_types.values})

    df['age_interval'] = df['age'].apply(age_interval)
    grouped_ages = df.groupby('age_interval')['successful_offer'].mean()
    grouped_ages_df = pd.DataFrame({'age': grouped_ages.index, '% of success': grouped_ages.values})

    df['income_interval'] = df['income'].apply(income_interval)
    incomes = df.groupby('income_interval')['successful_offer'].mean()
    incomes_df = pd.DataFrame({'income': incomes.index, '% of success': incomes.values})

    mean_success = df['successful_offer'].mean()

    graphs = [
        {
            'data': [
                Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=['#006341', '#d3d3d3']),
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )
            ],
            'layout': {
                'title': "Distribution of Offers' Success",
                'titlefont': {
                    'size': 24,
                    'color': '#006341'
                },
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white'
            }
        },

        {
            'data': [
                Bar(
                    x=offer_types_df['offer_type'],
                    y=offer_types_df['% of success'],
                    marker=dict(color='#006341')
                )
            ],
            'layout': {
                'title': 'Probability of Success by Offer Type',
                'titlefont': {
                    'size': 24,
                    'color': '#006341'
                },
                'yaxis': {
                    'title': "% of Success",
                    'titlefont': {
                        'size': 18,
                        'color': '#006341'
                    },
                    'tickfont': {
                        'size': 14
                    }
                },
                'xaxis': {
                    'title': "Offer Type",
                    'titlefont': {
                        'size': 18,
                        'color': '#006341'
                    },
                    'tickfont': {
                        'size': 14
                    }
                },
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'shapes': [
                    {
                        'type': 'line',
                        'x0': -0.5,
                        'x1': len(offer_types_df['offer_type']) - 0.5,
                        'y0': mean_success,
                        'y1': mean_success,
                        'line': {
                            'color': 'black',
                            'width': 2,
                            'dash': 'dash',
                        },
                    }
                ],
                'annotations': [
                    {
                        'x': 0,
                        'y': mean_success + 0.01,
                        'xref': 'x',
                        'yref': 'y',
                        'text': 'Mean % of success',
                        'showarrow': False,
                        'font': {
                            'size': 12,
                            'color': 'black'
                        }
                    }
                ]
            }
        },
        {
            'data': [
                Bar(
                    x=grouped_ages_df['age'],
                    y=grouped_ages_df['% of success'],
                    marker=dict(color='#006341')
                )
            ],
            'layout': {
                'title': 'Probability of Success by Age',
                'titlefont': {
                    'size': 24,
                    'color': '#006341'
                },
                'yaxis': {
                    'title': "% of Success",
                    'titlefont': {
                        'size': 18,
                        'color': '#006341'
                    },
                    'tickfont': {
                        'size': 14
                    }
                },
                'xaxis': {
                    'title': "Age",
                    'titlefont': {
                        'size': 18,
                        'color': '#006341'
                    },
                    'tickfont': {
                        'size': 14
                    }
                },
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'shapes': [
                    {
                        'type': 'line',
                        'x0': -0.5,
                        'x1': len(grouped_ages_df['age']) - 0.5,
                        'y0': mean_success,
                        'y1': mean_success,
                        'line': {
                            'color': 'black',
                            'width': 2,
                            'dash': 'dash',
                        },
                    }
                ],
                'annotations': [
                    {
                        'x': 0,
                        'y': mean_success + 0.01,
                        'xref': 'x',
                        'yref': 'y',
                        'text': 'Mean % of success',
                        'showarrow': False,
                        'font': {
                            'size': 12,
                            'color': 'black'
                        }
                    }
                ]
            }
        },
        {
            'data': [
                Bar(
                    x=incomes_df['income'],
                    y=incomes_df['% of success'],
                    marker=dict(color='#006341')
                )
            ],
            'layout': {
                'title': 'Probability of Success by Customer Income',
                'titlefont': {
                    'size': 24,
                    'color': '#006341'
                },
                'yaxis': {
                    'title': "% of Success",
                    'titlefont': {
                        'size': 18,
                        'color': '#006341'
                    },
                    'tickfont': {
                        'size': 14
                    }
                },
                'xaxis': {
                    'title': "Income (in thousands USD)",
                    'titlefont': {
                        'size': 18,
                        'color': '#006341'
                    },
                    'tickfont': {
                        'size': 14
                    }
                },
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'shapes': [
                    {
                        'type': 'line',
                        'x0': -0.5,
                        'x1': len(incomes_df['income']) - 0.5,
                        'y0': mean_success,
                        'y1': mean_success,
                        'line': {
                            'color': 'black',
                            'width': 2,
                            'dash': 'dash',
                        },
                    }
                ],
                'annotations': [
                    {
                        'x': 0,
                        'y': mean_success + 0.01,
                        'xref': 'x',
                        'yref': 'y',
                        'text': 'Mean % of success',
                        'showarrow': False,
                        'font': {
                            'size': 12,
                            'color': 'black'
                        }
                    }
                ]
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


class ChoiceForm(FlaskForm):
    user_ids = list(df['id'].unique())
    offer_ids = list(df['offer_id'].unique())
    user_id = SelectField('User ID', choices=list(zip(user_ids, user_ids)))
    offer_id = SelectField('Offer ID', choices=list(zip(offer_ids, offer_ids)))


@app.route('/machine-learning')
def machine_learning():
    """Machine learning page."""
    form = ChoiceForm()
    user_id_value = form.user_id.choices[0][0]
    offer_id_value = form.offer_id.choices[0][0]
    offer_values_dict, user_values_dict = get_offer_user_attributes(portfolio_after_engineer=portfolio,
                                                                        profile_after_engineered=profile,
                                                                        user_id=user_id_value,
                                                                        offer_id=offer_id_value)
    data = {'user_info': user_values_dict, 'offer_info': offer_values_dict}

    return render_template('machine-learning.html', data=data, form=form)


def convert_values_to_string(values):
    for key, value in values.items():
        values[key] = str(value)
    return values


@app.route('/update/<user_id>')
def update(user_id):
    """Update information about user ID."""
    form = ChoiceForm()
    _, user_values_dict = get_offer_user_attributes(portfolio_after_engineer=portfolio,
                                                        profile_after_engineered=profile,
                                                        user_id=user_id,
                                                        offer_id=form.offer_id.choices[0][0])

    data = {'user_info': convert_values_to_string(user_values_dict)}
    return jsonify(data)


@app.route('/update-offer/<offer_id>')
def update_offer(offer_id):
    """Update information about offer ID."""
    form = ChoiceForm()
    offer_values_dict, _ = get_offer_user_attributes(portfolio_after_engineer=portfolio,
                                                         profile_after_engineered=profile,
                                                         user_id=form.user_id.choices[0][0],
                                                         offer_id=offer_id)

    data = {'offer_info': convert_values_to_string(offer_values_dict)}
    return jsonify(data)


@app.route('/predict/<offer_id>/<user_id>')
def make_prediction(offer_id, user_id):
    """Make a prediction for the given offer ID and user ID."""
    offer_values_dict, user_values_dict = get_offer_user_attributes(portfolio_after_engineer=portfolio,
                                                                        profile_after_engineered=profile,
                                                                        user_id=user_id,
                                                                        offer_id=offer_id)
    
    prediction = int(predict(model = model, offer_values_dict=offer_values_dict, user_values_dict=user_values_dict))
    data = {'prediction': prediction}
    return jsonify(data)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()