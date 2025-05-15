import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np

# --- Import functions from the refactored visualization scripts ---
try:
    from visualize_toilet_new import get_toilet_data, create_toilet_figure
    print("Successfully imported toilet functions.")
except ImportError as e:
    print(f"Error importing from visualize_toilet_new.py: {e}")
    # Fournir des stubs qui retournent le bon nombre d'éléments attendus
    def get_toilet_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
    def create_toilet_figure(*args): return go.Figure().update_layout(title_text="Error loading toilet module")

try:
    from visualize_sleep_quiet_new import get_sleep_data, create_sleep_figure
    print("Successfully imported sleep functions.")
except ImportError as e:
    print(f"Error importing from visualize_sleep_quiet_new.py: {e}")
    def get_sleep_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_sleep_figure(*args): return go.Figure().update_layout(title_text="Error loading sleep module")

try:
    from visualize_outing_new import get_outings_data, create_outings_figure
    print("Successfully imported outings functions.")
except ImportError as e:
    print(f"Error importing from visualize_outing_new.py: {e}")
    def get_outings_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_outings_figure(*args): return go.Figure().update_layout(title_text="Error loading outings module")


# --- Configuration Globale ---
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'

TOILET_APP_NAME = "Toilet Activity"
SLEEP_APP_NAME = "Sleep Activity"
OUTINGS_APP_NAME = "Outings Activity"

# --- Chargement Initial des Données ---
print("Loading initial data...")
try:
    toilet_daily_data, toilet_monthly_data, toilet_failure_daily_markers = get_toilet_data()
    print(f"Toilet data loaded: Daily shape={toilet_daily_data.shape}, Monthly shape={toilet_monthly_data.shape}, Toilet Failures shape={toilet_failure_daily_markers.shape}")
except ValueError as ve: 
    print(f"Error unpacking data from get_toilet_data (expected 3 values, check visualize_toilet_new.py): {ve}")
    toilet_daily_data, toilet_monthly_data, toilet_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
except Exception as e:
    print(f"Error running get_toilet_data: {e}")
    toilet_daily_data, toilet_monthly_data, toilet_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

try:
    sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = get_sleep_data()
    print(f"Sleep data loaded: Daily shape={sleep_daily_data.shape}, Monthly shape={sleep_monthly_data.shape}, Sleep Failures shape={sleep_bed_failure_daily_markers.shape}")
except Exception as e:
    print(f"Error running get_sleep_data: {e}")
    sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

try:
    outings_daily_data, outings_monthly_data, outings_door_failure_daily_markers = get_outings_data()
    print(f"Outings data loaded: Daily shape={outings_daily_data.shape}, Monthly shape={outings_monthly_data.shape}, Door Failures shape={outings_door_failure_daily_markers.shape}")
except Exception as e:
    print(f"Error running get_outings_data: {e}")
    outings_daily_data, outings_monthly_data, outings_door_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

print("Initial data loading complete.")

# --- Initialisation de l'App Dash ---
app = Dash(__name__)
app.title = "Activity Dashboard Manager"

# --- Définition du Layout de l'App ---
app.layout = html.Div(id="app-container", children=[
    html.H2(app.title, id="app-title"),
    html.Div(id="control-container", children=[
        html.Div(className='selector', children=[
            html.Label("Activity :"),
            dcc.Dropdown(
                id='activity-type-selector',
                options=[
                    {'label': TOILET_APP_NAME, 'value': 'toilet'},
                    {'label': SLEEP_APP_NAME, 'value': 'sleep'},
                    {'label': OUTINGS_APP_NAME, 'value': 'outings'}
                ],
                value='toilet', 
                clearable=False,
                className='dash-dropdown'
            ),
        ]),
        html.Div(className='selector', children=[
            html.Label("View scale :"),
            dcc.RadioItems(
                id='scale-selector',
                options=[
                    {'label': 'Year View (Monthly)', 'value': 'year'},
                    {'label': 'Month View (Daily)', 'value': 'month'}
                ],
                value='year', 
                labelStyle={'display': 'inline-block', 'marginRight': '22px', 'color': TEXT_COLOR},
                inputStyle={'marginRight': '5px'}
            ),
        ]),
        html.Div(id='month-dropdown-container', className='selector', children=[
            html.Label(""),
            dcc.Dropdown(
                id='month-dropdown',
                clearable=False,
                className='dash-dropdown'
            )
        ], style={'display': 'none'}),
    ]),
    dcc.Graph(id='activity-graph')
])

# --- Callbacks ---
@app.callback(
    Output('month-dropdown-container', 'style'),
    Input('scale-selector', 'value')
)
def toggle_month_dropdown_visibility(scale):
    if scale == 'month':
        return {'display': 'flex'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('month-dropdown', 'options'),
    Output('month-dropdown', 'value'),
    Input('activity-type-selector', 'value')
)
def update_month_dropdown_options(selected_activity_type):
    options = []
    value = None
    df_daily_for_months = pd.DataFrame()
    df_failure_for_months = pd.DataFrame() 

    if selected_activity_type == 'toilet':
        df_daily_for_months = toilet_daily_data
        df_failure_for_months = toilet_failure_daily_markers 
    elif selected_activity_type == 'sleep':
        df_daily_for_months = sleep_daily_data
        df_failure_for_months = sleep_bed_failure_daily_markers
    elif selected_activity_type == 'outings':
        df_daily_for_months = outings_daily_data
        df_failure_for_months = outings_door_failure_daily_markers

    available_months_set = set()
    if not df_daily_for_months.empty and 'year_month' in df_daily_for_months.columns:
        available_months_set.update(df_daily_for_months['year_month'].dropna().unique())
    
    if not df_failure_for_months.empty and 'year_month' in df_failure_for_months.columns:
        available_months_set.update(df_failure_for_months['year_month'].dropna().unique())

    if available_months_set:
        available_months_sorted = sorted(list(available_months_set))
        options = [{'label': m, 'value': m} for m in available_months_sorted]
        if available_months_sorted:
            value = available_months_sorted[0] # Sélectionner le premier mois par défaut
    return options, value


@app.callback(
    Output('activity-graph', 'figure'),
    Input('activity-type-selector', 'value'),
    Input('scale-selector', 'value'),
    Input('month-dropdown', 'value')
)
def update_main_graph(activity_type, scale, selected_month):
    fig = go.Figure()
    try:
        if activity_type == 'toilet':
      
            fig = create_toilet_figure(
                toilet_daily_data,
                toilet_monthly_data,
                toilet_failure_daily_markers, 
                scale,
                selected_month
            )
        elif activity_type == 'sleep':
            fig = create_sleep_figure(
                sleep_daily_data,
                sleep_monthly_data,
                sleep_bed_failure_daily_markers,
                scale,
                selected_month
            )
        elif activity_type == 'outings':
            fig = create_outings_figure(
                outings_daily_data,
                outings_monthly_data,
                outings_door_failure_daily_markers, 
                scale,
                selected_month
            )
        else:
            fig.update_layout(title=dict(text="Select an activity type", font=dict(color=TEXT_COLOR)))
    except Exception as e:
        print(f"Error creating figure for {activity_type}: {e}")
        # Afficher une figure d'erreur si la création du graphique échoue
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            title=dict(text=f"Error generating graph for {activity_type}: Check console", font=dict(color='red'))
        )

    # Assurer un état par défaut si aucune donnée n'est tracée et aucun titre d'erreur n'est défini
    if not fig.data and (not fig.layout or not fig.layout.title or not fig.layout.title.text):
        fig.update_layout(
            template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            title=dict(text="Veuillez sélectionner une activité et une échelle.", font=dict(color=TEXT_COLOR))
        )
    return fig

# --- Exécution de l'App ---
if __name__ == '__main__':
    print("Starting Dash Manager App...")
    app.run(debug=True)