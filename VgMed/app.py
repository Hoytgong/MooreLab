# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
LogisticRegression(solver='lbfgs')


import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import io
import requests


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions']=True

# Setting up the data infrastructure
url = "https://raw.githubusercontent.com/Hoytgong/vgmed/master/finalBCdata.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
df_id = df.drop('phenotype', axis=1)
features = df_id.drop('patient_id', axis=1).values

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, df['phenotype'].values, random_state=42)

exported_pipeline = ExtraTreesClassifier(bootstrap=False, random_state = 42, criterion="entropy", max_features=0.1, min_samples_leaf=18, min_samples_split=20, n_estimators=100)
exported_pipeline.fit(training_features, training_target)

testing_features_cancer = testing_features[testing_target == 1,:]
testing_features_healthy = testing_features[testing_target != 1,:]

prob_healthy = exported_pipeline.predict_proba(testing_features_healthy)[:,1]
prob_cancer = exported_pipeline.predict_proba(testing_features_cancer)[:,1]

hist_data = [prob_healthy, prob_cancer]
group_labels = ['Healthy', 'Breast Cancer']
colors = ['#3A4750', '#F64E8B']

## Actual GUI
app.layout = html.Div(id='test', children=[
    html.H1('VGMED'),
    html.H6('Predictive Gene Editing Variant Dashboard'),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-data-upload'),


    html.Label('Subject ID'),
    dcc.Dropdown(id='subject-id',
                 options=[{'label': i, 'value': i} for i in df['patient_id'].values.tolist()]),
    dcc.RadioItems(
        id='editing-choice',
        options=[{'label': i, 'value': i} for i in ['Edit SNP', 'Optimize SNP']]
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),

    dcc.Graph(id='population-distribution-graph'),

    html.Div(id='branch')

])


###### Defining Callback methods #####

## Distribution Figure
@app.callback(
    Output('population-distribution-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('subject-id', 'value')])
def update_graph(clicks, sub_id):
    x_row = df_id.loc[df_id.patient_id == int(sub_id), :]
    x_features = x_row.drop('patient_id', axis=1).values
    x_pos = exported_pipeline.predict_proba(x_features)[:, 1][0]

    fig = ff.create_distplot(hist_data, group_labels, bin_size=.35, curve_type='normal', show_hist=False, colors=colors)
    fig['layout'].update(title='Risk Score Distribution of All Patients')
    fig['layout'].update(shapes=[{'type': 'line', 'x0': x_pos, 'y0': 0, 'x1': x_pos, 'y1': 22,
                                  'line': {'color': '#F64E8B','width': 2}, }])
    return fig


## Defining the parse ability
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


## Branched callback to nest children of either Optimize SNP or Edit SNP
@app.callback(
    Output('branch', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('subject-id', 'value'),
     State('editing-choice', 'value')])
def return_optimized(clicks, sub_id, editing_choice):
    if editing_choice == 'Optimize SNP':
        dff = optimization_table(int(sub_id))
        table = ff.create_table(dff, height_constant=10)
        return html.Div([dcc.Graph(id='table', figure=table)
        ])
    elif editing_choice == 'Edit SNP':
        return html.Div([
            html.Label('Select SNP to modify'),
            dcc.Dropdown(
                id='SNP-dropdown',
                options=[{'label': i, 'value': i} for i in list(df.columns.values)[1:-1]]
            ),
            html.Div(id='text'),

            html.Label('Edit SNP to...'),
            dcc.Dropdown(id='SNP-values'),

            html.Button(id='indiv-submit', n_clicks=0, children='Edit'),
            html.Div(id='hidden')
        ])
    else:
        return -1


## Individual Editing steps
@app.callback(
    Output('text', 'children'),
    [Input('subject-id', 'value'), Input('SNP-dropdown', 'value')])
def return_curr_SNP_value(sub_id, SNP):
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    value = df.loc[df.patient_id == int(sub_id), str(SNP)].values[0]
    return 'Your current SNP value is "{}"'.format(value)


@app.callback(
    Output('SNP-values', 'options'),
    [Input('subject-id', 'value'),
     Input('SNP-dropdown', 'value')])
def return_editing_options(sub_id, SNP):
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    list = [0, 1, 2]
    value = df.loc[df.patient_id == int(sub_id), str(SNP)].values[0]
    list.remove(value)
    return [{'label': i, 'value': i} for i in list]

@app.callback(
    Output('hidden', 'children'),
    [Input('indiv-submit', 'n_clicks')],
    [State('subject-id', 'value'),
     State('SNP-dropdown', 'value'),
     State('SNP-values', 'value')])
def individual_editing(in_clicks, sub_id, SNP, new_SNP_val):
    if in_clicks > 0:
        indiv_df = individual_editing(sub_id, SNP, new_SNP_val)
        indiv_table = ff.create_table(indiv_df, height_constant=10)
        return html.Div([dcc.Graph(id='indiv-table', figure=indiv_table)])

#Obtaining beta values
listofbetas = []
for i in range(training_features.shape[1]):
    logit = LogisticRegression()
    logit.fit(training_features, training_target)
    beta = logit.coef_.flatten()[i]
    listofbetas.append(beta)
beta_matrix = np.matrix(listofbetas).transpose()


## Put Logit regressor here
def individual_editing(sub_id, SNP, new_SNP_val):
    all_info = []
    x_row = df_id.loc[df_id.patient_id == int(sub_id), :]
    x_features = x_row.drop('patient_id', axis=1).values

    original_prob = exported_pipeline.predict_proba(x_features)[:, 1][0] ############
    og_risk_score = x_features * beta_matrix
    og_risk_score = np.array(og_risk_score)[0][0]


    ori_val = x_row.loc[:, SNP].values[0]

    x_row.loc[:, SNP].values[0] = new_SNP_val
    new_x_features = x_row.drop('patient_id', axis=1).values
    new_prob = exported_pipeline.predict_proba(new_x_features)[:, 1][0] #########
    new_risk_score = new_x_features * beta_matrix
    new_risk_score = np.array(new_risk_score)[0][0]

    all_info.append(sub_id)
    all_info.extend((og_risk_score, new_risk_score, og_risk_score - new_risk_score))
    all_info.extend((SNP, ori_val, new_SNP_val))

    final_df = pd.DataFrame(all_info).T
    final_df.columns = ['Participant Idx', 'Original Risk Score', 'New Risk Score', 'Risk Delta', 'SNP Name',
                        'Ori. SNP Value', 'New SNP Value']
    return final_df


## Function defining optimization table, individual editing works, not optimization

## Put Logit regressor here
def optimization_table(sub_id):
    SNP_names = df.columns.values.tolist()[1:-1]
    n_snps = len(SNP_names)

    all_info = []
    x_row = df_id.loc[df_id.patient_id == int(sub_id), :]
    x_features = x_row.drop('patient_id', axis=1).values
    # original_prob = exported_pipeline.predict_proba(x_features)[:, 1][0] ##########
    og_risk_score = x_features * beta_matrix
    og_risk_score = np.array(og_risk_score)[0][0]

    mylistup = []
    mylistdown = []

    for snp in range(0, x_features.size):
        x_features[0][snp] = (x_features[0][snp] + 1) % 3
        scoreup = x_features * beta_matrix
        scoreup = np.array(scoreup)[0][0]
        mylistup.append(scoreup)

        x_features[0][snp] = (x_features[0][snp] - 2) % 3  # -2+1=-1
        scoredown = x_features * beta_matrix
        scoredown = np.array(scoredown)[0][0]
        mylistdown.append(scoredown)

        x_features[0][snp] = (x_features[0][snp] + 1) % 3  # change back

    # for snp in range(0, x_features.size):
    #     x_features[0][snp] = (x_features[0][snp] + 1) % 3
    #     probup = exported_pipeline.predict_proba(x_features.reshape(1, -1))[0, 1] ######
    #     mylistup.append(probup)
    #
    #     x_features[0][snp] = (x_features[0][snp] - 2) % 3  # -2+1=-1
    #     probdown = exported_pipeline.predict_proba(x_features.reshape(1, -1))[0, 1] #######
    #     mylistdown.append(probdown)
    #
    #     x_features[0][snp] = (x_features[0][snp] + 1) % 3  # change back

    upbumplist = mylistup - og_risk_score
    downbumplist = mylistdown - og_risk_score
    uplist = upbumplist.tolist()
    downlist = downbumplist.tolist()

    completelist = uplist + downlist + [0]  # 0 for no change
    min_index = np.argmin(completelist)
    min_value = min(completelist)
    best_score = min_value + og_risk_score
    all_info.append(sub_id)  # participant idx

    all_info.extend((og_risk_score, best_score, og_risk_score - best_score)) #######
    if (min_index == (n_snps * 2 + 1)):  # no change is best
        all_info.extend(("N/A", "N/A", "N/A"))
    else:
        SNP_idx = min_index % n_snps
        all_info.extend((SNP_names[SNP_idx], int(x_features[0][SNP_idx]),
                         int((x_features[0][SNP_idx] - 1 + 2 * float(min_index <= n_snps)) % 3)))

    final_df = pd.DataFrame(all_info).T
    final_df.columns = ['Participant Idx', 'Original Risk Score', 'Best Possible Score', 'Risk Decrease', 'SNP Name',
                        'Ori. SNP Value', 'New SNP Value']
    return final_df

if __name__ == '__main__':
    app.run_server(debug=True)
