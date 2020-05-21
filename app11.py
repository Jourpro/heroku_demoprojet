from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from werkzeug.serving import run_simple
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import preprocessing

external_stylesheets = ['/static/bootstrap.min.css']

server = flask.Flask(__name__)
dash_app1 = Dash(__name__, server = server, url_base_pathname='/dashboard/',external_stylesheets=external_stylesheets)


df1 = pd.read_csv('df_test_red_dash.csv')
#df = df1
df = df1[:200]


std_scale_red = preprocessing.StandardScaler()
X_red_scaled = df.drop(columns=['SK_ID_CURR'])
X_red_scaled[X_red_scaled.columns] = std_scale_red.fit_transform(X_red_scaled[X_red_scaled.columns])

model_xgb = xgb.XGBClassifier()
booster = xgb.Booster()
booster.load_model('xgb_forflask.xgb')
model_xgb._Booster = booster

available_indicators = (np.array(df.drop(columns=['SK_ID_CURR']).columns))
flat_indicators = np.array(df[['Residency external option rating', 'Client region with city rating']].columns)
credit_indicators= np.array(df[['Cash or revolving loan', 'Number of drawings for a month, 60 days ago', 'Number of paid installment', 'Days past due during the month of previous credit', 'Total years of loan (loan/annuity)', 'Prescribed installment amount of previous credit on this installment']].columns)
sk_indicators = df['SK_ID_CURR']


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

dash_app1.layout = html.Div([


    html.H2(children='Loan survey for the client number: ', style={
            'textAlign': 'center',
            'color': colors['text']}),

    html.Div([  
        ],style={'width': '33%', 'display': 'inline-block'}),
 
    html.Div([
         
            dcc.Dropdown(
                id='Client number',
                options=[{'label': i, 'value': i} for i in sk_indicators],
                value=sk_indicators[0],
                style={"width": "100%","verticalAlign":"middle"})   
        ],style={'width': '33%','textAlign': 'center', 'display': 'inline-block'}),


    html.Div([  
        ],style={'width': '33%', 'display': 'inline-block'}),
#,style=dict(display='flex')
#"justify-content": "center"
#style = {"width":"100%", "display": "flex", "align-items": "center"}

    html.Div(id='display-selected-values',style={'font-weight': 'bold'}),
    #, 'font-style': 'italic', 'font-weight': 'bold',color': 'blue', 
    

    html.H3(children='Financial behavior, Real Estate type and Working exprience and education type for all the clients', style={
        'textAlign': 'center',
        'color': colors['text']
        }),


#    html.Div(id='display-selected-values'),

    html.Div([


        html.Div([
    #       html.Label(['Real Estate, Car Owner, Contract type and Education type']),
            dcc.Dropdown(
                id='my_dropdown_binaries',
                options=[
                        {'label': 'Cash or revolving loan', 'value': 'Cash or revolving loan'},
                        {'label': 'Number of drawings for a month, 60 days ago', 'value': 'Number of drawings for a month, 60 days ago'},
                        {'label': 'Number of paid installment', 'value': 'Number of paid installment'},
                        {'label': 'Days past due during the month of previous credit', 'value': 'Days past due during the month of previous credit'},
                ],
                value='Number of paid installment',
                multi=False,
                clearable=False
            ),
        ],
        style={'width': '33%', 'display': 'inline-block'}
    #    className='col-4'
    #    className = 'four columns'
        ),

        html.Div([
            dcc.Dropdown(
                id='my_dropdown_binaries2',
                options=[
                        {'label': 'Residency external option rating', 'value': 'Residency external option rating'},
                        {'label': 'Client region with city rating', 'value': 'Client region with city rating'},
                ],
                value='Client region with city rating',
                multi=False,
                clearable=False
            ),
        ],
        style={'width': '33%', 'display': 'inline-block'}
    #    className='col-4'
    #    className = 'four columns'
     ),
        

        html.Div([
            dcc.Dropdown(
                id='my_dropdown_binaries3',
                options=[
                        {'label': 'Education income rating', 'value': 'Education income rating'},
                        {'label': 'Age, employment experience, registration and publication date rating', 'value': 'Age, employment experience, registration and publication date rating'},
                ],
                value='Education income rating',
                multi=False,
                clearable=False
            ),
        ],
        style={'width': '33%', 'display': 'inline-block'}
    #    className='col-4'
    #     className = 'four columns'
        ),

        ],
        className='row'
    ),
#    html.Div([
#        dcc.Graph(id='the_graph_binaries',style={"width": "25%"}),
#        dcc.Graph(id='the_graph_binaries2',style={"width": "25%"})
#    ]),

    html.Div([
        dcc.Graph(id='the_graph_binaries'),
    ], style={'display': 'inline-block', 'width': '33%'}),

    html.Div([
        dcc.Graph(id='the_graph_binaries2'),
    ], style={'display': 'inline-block', 'width': '33%'}),

    html.Div([
        dcc.Graph(id='the_graph_binaries3'),
    ], style={'display': 'inline-block', 'width': '33%'}),


    html.Div([

        html.Div([
            dcc.Dropdown(
                id='dropdown_hist1',
                options=[{'label': i, 'value': i} for i in flat_indicators],
                value='Residency external option rating'
            ),
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='dropdown_hist2',
                options=[{'label': i, 'value': i} for i in credit_indicators],
                value='Total years of loan (loan/annuity)'
                ),
        ],
        style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
    ]),

    html.Div([
        dcc.Graph(id='the_graph_hist1'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='the_graph_hist2'),
    ], style={'display': 'inline-block', 'width': '49%'}),

 #   html.Div([
 #       dcc.Graph(id='the_graph_binaries2',style={"width": "25%"})
 #   ]),
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Total years of loan (loan/annuity)'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Prescribed installment amount of previous credit on this installment'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div([  
        ],style={'width': '9%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='indicator-graphic'),

        html.Div(children='Client region with city rating', style={'textAlign': 'left',
            'color': colors['text']}),

        dcc.Slider(
            id='crr-slider',
            min=df['Client region with city rating'].min(),
            max=df['Client region with city rating'].max(),
            value=df['Client region with city rating'].max(),
            marks={str(ccr): str(ccr) for ccr in df['Client region with city rating'].unique()},
            step=None,
        ),

        ], style={'display': 'inline-block', 'width': '82%'}),

    html.Div([  
        ],style={'width': '9%', 'display': 'inline-block'})
    
],
className='container')

@dash_app1.callback(
    Output('display-selected-values', 'children'),
    [Input('Client number', 'value')])

#def set_display_children(client_sk_num):
#    in_dat = int(df['OWN_CAR_VALUABLE'][df['SK_ID_CURR'] == client_sk_num].iloc[0])
#    return u'The client number {} {} '.format(client_sk_num,int(df['OWN_CAR_VALUABLE'][df['SK_ID_CURR'] == client_sk_num].iloc[0]),)

def set_display_children(client_sk_num):
    i1 = df[df['SK_ID_CURR'] == client_sk_num].index
    df_sk_id_select = pd.DataFrame(data =X_red_scaled.iloc[i1].values.reshape(1, -1), columns=X_red_scaled.columns)
    target = model_xgb.predict_proba(df_sk_id_select)[:, 1] 
    if target >= 0.04:
        target_message='ALLOWED'
    elif target < 0.04:
        target_message='REJECTED'

    a1 = int(df['Client region with city rating'][df['SK_ID_CURR'] == client_sk_num].iloc[0])
    a2 = int(df['Education income rating'][df['SK_ID_CURR'] == client_sk_num].iloc[0])
    a3 = int(df['Cash or revolving loan'][df['SK_ID_CURR'] == client_sk_num].iloc[0])
    if a1 == 1:
        city_rating = 'first'
    elif a1 == 2:
        city_rating = 'second' 
    elif a1 == 3:
        city_rating = 'third'
    
    if a2 == 2:
        education_income_rating = 'high'
    elif a2 == 1:
        education_income_rating = 'middle' 
    elif a2 == 0:
        education_income_rating = 'low'
    
    if a3 == 1:
        cash_revolving_loan = 'cash loan' 
    elif a3 == 0:
        cash_revolving_loan = 'revolving loan'
#Cash or revolving loan
    return (u'The client number {} has been {} to loan based on the following criteria \
        in order of significance. The client has {} total years of loan (loan/annuity) and \
            ${} of prescribed installment amount of previous credit on this installment, \
                 {:.2f} as residency external option rating (best score 1), and {}  paid \
                     installments. The client had {} drawings for a month, 60 days ago \
                         and {} days past due during the month of previous credit. The \
                              client region with city rating is {}. The education income \
                                   rating is {}. This is a {}.'.format(client_sk_num,target_message,
    int(df['Total years of loan (loan/annuity)'][df['SK_ID_CURR'] == client_sk_num].iloc[0]),
    int(df['Prescribed installment amount of previous credit on this installment'][df['SK_ID_CURR'] == client_sk_num].iloc[0]),
    df['Residency external option rating'][df['SK_ID_CURR'] == client_sk_num].iloc[0],
    int(df['Number of paid installment'][df['SK_ID_CURR'] == client_sk_num].iloc[0]),
    int(df['Number of drawings for a month, 60 days ago'][df['SK_ID_CURR'] == client_sk_num].iloc[0]),
    int(df['Days past due during the month of previous credit'][df['SK_ID_CURR'] == client_sk_num].iloc[0]),
    city_rating,
    education_income_rating,
    cash_revolving_loan))

@dash_app1.callback(
    Output(component_id='the_graph_binaries', component_property='figure'),
    [Input(component_id='my_dropdown_binaries', component_property='value')])

def update_graph1(my_dropdown_b):
    dff_b = (df[['Cash or revolving loan',
    'Number of drawings for a month, 60 days ago',
    'Number of paid installment',
    'Days past due during the month of previous credit']])

    dff_b.loc[df['Cash or revolving loan'] == 1, 'Cash or revolving loan'] = 'Cash loan (1)'
    dff_b.loc[df['Cash or revolving loan'] == 0, 'Cash or revolving loan'] = 'Revolving loan (0)'

    dff_b.loc[df['Number of drawings for a month, 60 days ago'] > 39, 'Number of drawings for a month, 60 days ago'] = '> 39'
    dff_b.loc[(df['Number of drawings for a month, 60 days ago'] < 40) & (df['Number of drawings for a month, 60 days ago'] > 29) , 'Number of drawings for a month, 60 days ago'] = '40 > x > 29'
    dff_b.loc[(df['Number of drawings for a month, 60 days ago'] < 30) & (df['Number of drawings for a month, 60 days ago'] > 19) , 'Number of drawings for a month, 60 days ago'] = '30> x > 19'
    dff_b.loc[(df['Number of drawings for a month, 60 days ago'] < 20) & (df['Number of drawings for a month, 60 days ago'] > 9) , 'Number of drawings for a month, 60 days ago'] = '20 > x > 9'
    dff_b.loc[df['Number of drawings for a month, 60 days ago'] < 10, 'Number of drawings for a month, 60 days ago'] = '< 10'

    dff_b.loc[df['Number of paid installment'] > 99, 'Number of paid installment'] = '> 99'
    dff_b.loc[(df['Number of paid installment'] < 100) & (df['Number of paid installment'] > 74) , 'Number of paid installment'] = '100 > x > 74'
    dff_b.loc[(df['Number of paid installment'] < 75) & (df['Number of paid installment'] > 49) , 'Number of paid installment'] = '75> x > 49'
    dff_b.loc[(df['Number of paid installment'] < 50) & (df['Number of paid installment'] > 24) , 'Number of paid installment'] = '50 > x > 24'
    dff_b.loc[df['Number of paid installment'] < 25, 'Number of paid installment'] = '< 25'

    dff_b.loc[df['Days past due during the month of previous credit'] > 0, 'Days past due during the month of previous credit'] = '> 0'
    dff_b.loc[df['Days past due during the month of previous credit'] == 0, 'Days past due during the month of previous credit'] = '= 0'
#           
#    print(df.iloc[1])

    piechart=px.pie(
            data_frame=dff_b,
            names=my_dropdown_b,
            hole=.1,
            )

    return (piechart)

@dash_app1.callback(
    Output(component_id='the_graph_binaries2', component_property='figure'),
    [Input(component_id='my_dropdown_binaries2', component_property='value')])


def update_graph1_2(my_dropdown_b_2):
    dff_b = df[['Residency external option rating','Client region with city rating']]
    
    dff_b.loc[df['Client region with city rating'] == 1, 'Client region with city rating'] = 'First'
    dff_b.loc[df['Client region with city rating'] == 2, 'Client region with city rating'] = 'Second'
    dff_b.loc[df['Client region with city rating'] == 3, 'Client region with city rating'] = 'Third'
    
    dff_b.loc[df['Residency external option rating'] >= 0.115, 'Residency external option rating'] = '> 0.115'
    dff_b.loc[(df['Residency external option rating'] < 0.115) & (df['Residency external option rating'] >= 0.1) , 'Residency external option rating'] = '0.115 > x > 0.1'
    dff_b.loc[(df['Residency external option rating'] < 0.1) & (df['Residency external option rating'] >= 0.085) , 'Residency external option rating'] = '0.1 > x > 0.085'
    dff_b.loc[(df['Residency external option rating'] < 0.085) & (df['Residency external option rating'] >= 0.06) , 'Residency external option rating'] = '0.085> x > 0.06'
    dff_b.loc[df['Residency external option rating'] < 0.06, 'Residency external option rating'] = '< 0.06'

    piechart_2=px.pie(
            data_frame=dff_b,
            names=my_dropdown_b_2,
            hole=.1,
            )

    return (piechart_2)

@dash_app1.callback(
    Output(component_id='the_graph_binaries3', component_property='figure'),
    [Input(component_id='my_dropdown_binaries3', component_property='value')])

def update_graph1_3(my_dropdown_b_3):
    dff_b = df[['Education income rating','Age, employment experience, registration and publication date rating']]
    
    dff_b.loc[df['Education income rating'] == 2, 'Education income rating'] = 'High (2 points)'
    dff_b.loc[df['Education income rating'] == 1, 'Education income rating'] = 'Middle (1 point)'
    dff_b.loc[df['Education income rating'] == 0, 'Education income rating'] = 'Low (0 points)'
    
    dff_b.loc[df['Age, employment experience, registration and publication date rating'] == 1, 'Age, employment experience, registration and publication date rating'] = 'High (1 point)'
    dff_b.loc[df['Age, employment experience, registration and publication date rating'] == 0, 'Age, employment experience, registration and publication date rating'] = 'Low (0 points)'

    piechart_3=px.pie(
            data_frame=dff_b,
            names=my_dropdown_b_3,
            hole=.1,
            )

    return (piechart_3)

@dash_app1.callback(
    Output(component_id='the_graph_hist1', component_property='figure'),
    [Input(component_id='dropdown_hist1', component_property='value')])

def update_hist1(my_dropdown_h1):
    dff_h1 = df.drop(columns=['SK_ID_CURR'])

    hist1chart=px.histogram(
            data_frame=dff_h1,
            x=my_dropdown_h1,
            nbins=5,
            )

    return (hist1chart)

@dash_app1.callback(
    Output(component_id='the_graph_hist2', component_property='figure'),
    [Input(component_id='dropdown_hist2', component_property='value')])

def update_hist2(my_dropdown_h2):
    dff_h2 = df.drop(columns=['SK_ID_CURR'])

    hist2chart=px.histogram(
            data_frame=dff_h2,
            x=my_dropdown_h2,
            nbins=5,
            )

    return (hist2chart)

@dash_app1.callback(Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value'),
    Input('crr-slider', 'value')])

def update_graph2(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 region_value):
    dff = df[df['Client region with city rating'] == region_value]

    return {
        'data': [dict(
            x=dff[xaxis_column_name],
            y=dff[yaxis_column_name],
            text=dff['SK_ID_CURR'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'titlefont':dict(size=12),
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 100, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest'
        )
    }


@server.route('/')
@server.route('/hello')
def hello():
    return 'Welcome to the demo project'

@server.route('/dashboard/')
def render_dashboard():
    return flask.redirect('/dash1')


app11 = DispatcherMiddleware(server, {
    '/dash1': dash_app1.server
})

#if __name__ == '__main__':
#    run_simple('localhost', 8050, app11) 

if __name__ == '__main__':
    server.run(debug=True)