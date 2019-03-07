# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

import plotly.figure_factory as ff
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True

# Setting up the data infrastructure
df = pd.read_csv('finalBCdata.csv')
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


app.layout = html.Div(children=[
    html.H1('VGMED'),
    html.Div('Predictive Gene Editing Variant Dashboard'),

    html.Label('Subject ID input'),
    dcc.Input(id='subject-id', type='text', value='83811'),
    dcc.RadioItems(
        id='editing-choice',
        options=[{'label': i, 'value': i} for i in ['Optimize','Individual Editing']]
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),

    dcc.Graph(id='population-distribution-graph'),

    html.Div(id='branch')

])

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

@app.callback(
    Output('branch', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('subject-id', 'value'),
     State('editing-choice', 'value')])
def return_optimized(clicks, sub_id, editing_choice):
    if editing_choice == 'Optimize':
        dff = optimization_table(sub_id)
        table = ff.create_table(dff)
        return html.Div([dcc.Graph(id='table', figure=table)
        ])
    elif editing_choice == 'Individual Editing':
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

@app.callback(
    Output('text', 'children'),
    [Input('subject-id', 'value'), Input('SNP-dropdown', 'value')])
def return_curr_SNP_value(sub_id, SNP):
    value = df.loc[df.patient_id == int(sub_id), str(SNP)][0]
    return 'Your current SNP value is "{}"'.format(value)

@app.callback(
    Output('SNP-values', 'options'),
    [Input('subject-id', 'value'),
     Input('SNP-dropdown', 'value')])
def return_editing_options(sub_id, SNP):
    df = pd.read_csv('finalBCdata.csv')
    list = [0, 1, 2]
    value = df.loc[df.patient_id == int(sub_id), str(SNP)][0]
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
        indiv_table = ff.create_table(indiv_df)
        return html.Div([dcc.Graph(id='indiv-table', figure=indiv_table)])


def individual_editing(sub_id, SNP, new_SNP_val):
    all_info = []
    x_row = df_id.loc[df_id.patient_id == int(sub_id), :]
    x_features = x_row.drop('patient_id', axis=1).values
    original_prob = exported_pipeline.predict_proba(x_features)[:, 1][0]
    ori_val = x_row.loc[:, SNP].values[0]

    x_row.loc[:, SNP].values[0] = new_SNP_val
    new_x_features = x_row.drop('patient_id', axis=1).values
    new_prob = exported_pipeline.predict_proba(new_x_features)[:, 1][0]

    all_info.append(sub_id)
    all_info.extend((original_prob, new_prob, original_prob - new_prob))
    all_info.extend((SNP, ori_val, new_SNP_val))

    final_df = pd.DataFrame(all_info).T
    final_df.columns = ['Participant Idx', 'Original Prob.', 'New Prob.', 'Risk Delta', 'SNP Name',
                        'Ori. SNP Value', 'New SNP Value']
    return final_df


def optimization_table(sub_id):
    SNP_names = df.columns.values.tolist()[1:-1]
    n_snps = len(SNP_names)

    all_info = []
    x_row = df_id.loc[df_id.patient_id == int(sub_id), :]
    x_features = x_row.drop('patient_id', axis=1).values
    original_prob = exported_pipeline.predict_proba(x_features)[:, 1][0]

    mylistup = []
    mylistdown = []

    for snp in range(0, x_features.size):
        x_features[0][snp] = (x_features[0][snp] + 1) % 3
        probup = exported_pipeline.predict_proba(x_features.reshape(1, -1))[0, 1]
        mylistup.append(probup)

        x_features[0][snp] = (x_features[0][snp] - 2) % 3  # -2+1=-1
        probdown = exported_pipeline.predict_proba(x_features.reshape(1, -1))[0, 1]
        mylistdown.append(probdown)

        x_features[0][snp] = (x_features[0][snp] + 1) % 3  # change back

    upbumplist = mylistup - original_prob
    downbumplist = mylistdown - original_prob
    uplist = upbumplist.tolist()
    downlist = downbumplist.tolist()

    completelist = uplist + downlist + [0]  # 0 for no change
    min_index = np.argmin(completelist)
    min_value = min(completelist)
    best_prob = min_value + original_prob
    all_info.append(sub_id)  # participant idx

    all_info.extend((original_prob, best_prob, original_prob - best_prob))
    if (min_index == (n_snps * 2 + 1)):  # no change is best
        all_info.extend(("N/A", "N/A", "N/A"))
    else:
        SNP_idx = min_index % n_snps
        all_info.extend((SNP_names[SNP_idx], int(x_features[0][SNP_idx]),
                         int((x_features[0][SNP_idx] - 1 + 2 * float(min_index <= n_snps)) % 3)))

    final_df = pd.DataFrame(all_info).T
    final_df.columns = ['Participant Idx', 'Original Prob.', 'Best Possible Prob.', 'Risk Decrease', 'SNP Name',
                        'Ori. SNP Value', 'New SNP Value']
    return final_df

if __name__ == '__main__':
    app.run_server(debug=True)
