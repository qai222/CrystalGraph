import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output, no_update

DF_FEAT = pd.read_csv("../data_formatted/data_tab_feat.csv")
DF_TARGET = pd.read_csv("../data_formatted/data_tab_target.csv")
# assert [i for i in DF_FEAT.index] == [j for j in DF_TARGET.index]

DF_FEAT = DF_FEAT.iloc[:, 1:]
DF_TARGET = DF_TARGET.iloc[:, 1:]

FEATURE_NAMES = DF_FEAT.columns.tolist()
TARGET_NAMES = DF_TARGET.columns.tolist()

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

app = Dash()
app.layout = html.Div([
    html.H5("FEATURE NAMES"),
    dcc.Dropdown(FEATURE_NAMES, [FEATURE_NAMES[0], ], id='dropdown_feat', multi=True),
    html.H5("TARGET NAMES"),
    dcc.Dropdown(TARGET_NAMES, [TARGET_NAMES[0], ], id='dropdown_target', multi=True),
    html.H5(),
    dcc.Graph(id="graph"),
])


@callback(
    Output('graph', 'figure'),
    Input('dropdown_feat', 'value'),
    Input('dropdown_target', 'value'),
)
def update_output(feat_names, target_names):
    fig = go.Figure()
    if len(feat_names) == 1 and len(target_names) >= 1:
        x = DF_FEAT[feat_names[0]]
        ys = target_names
        y = DF_TARGET
        xaxis_title = dict(text=feat_names[0], font=dict(size=16, color='#FFFFFF'))
        yaxis_title = dict(text="target 3D property", font=dict(size=16, color='#FFFFFF'))
    elif len(feat_names) >= 1 and len(target_names) == 1:
        x = DF_TARGET[target_names[0]]
        ys = feat_names
        y = DF_FEAT
        xaxis_title = dict(text=target_names[0], font=dict(size=16, color='#FFFFFF'))
        yaxis_title = dict(text="2D feature", font=dict(size=16, color='#FFFFFF'))
    elif len(feat_names) >= 1 and len(target_names) == 0:
        x = DF_FEAT
        xs = feat_names
        ys = None
        xaxis_title = dict(text="Selected Features", font=dict(size=16, color='#FFFFFF'))
        yaxis_title = dict(text="Count", font=dict(size=16, color='#FFFFFF'))
    elif len(target_names) >= 1 and len(feat_names) == 0:
        x = DF_TARGET
        xs = target_names
        ys = None
        xaxis_title = dict(text="Selected Targets", font=dict(size=16, color='#FFFFFF'))
        yaxis_title = dict(text="Count", font=dict(size=16, color='#FFFFFF'))
    else:
        return no_update

    if ys is None:
        for i, name in enumerate(xs):
            color = DEFAULT_PLOTLY_COLORS[i].replace(")", ", 0.8)").replace("rgb", "rgba")
            fig.add_trace(go.Histogram(x=x[name], opacity=0.5, marker=dict(color=color), name=name))
    else:
        for i, name in enumerate(ys):
            color = DEFAULT_PLOTLY_COLORS[i].replace(")", ", 0.2)").replace("rgb", "rgba")
            fig.add_trace(go.Scattergl(x=x, y=y[name], mode='markers', marker=dict(color=color, ), name=name))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)
