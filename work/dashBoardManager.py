import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import numpy as np

# --- Import functions from the refactored visualization scripts ---
# Ensure these files are in the same directory or accessible via Python's path
try:
    from visualize_toilet_new import get_toilet_data, create_toilet_figure
    print("Successfully imported toilet functions.")
except ImportError as e:
    print(f"Error importing from visualize_toilet_new.py: {e}")
    # Define dummy functions if import fails to prevent NameErrors later
    def get_toilet_data(): return pd.DataFrame(), pd.DataFrame()
    def create_toilet_figure(*args): return go.Figure().update_layout(title_text="Error loading toilet module")

try:
    from visualize_sleep_quiet_new import get_sleep_data, create_sleep_figure
    print("Successfully imported sleep functions.")
except ImportError as e:
    print(f"Error importing from visualize_sleep_quiet_new.py: {e}")
    # Define dummy functions if import fails
    def get_sleep_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_sleep_figure(*args): return go.Figure().update_layout(title_text="Error loading sleep module")


# --- Configuration Globale ---
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
TOILET_APP_NAME = "Toilet Activity"
SLEEP_APP_NAME = "Sleep Activity"

# --- Chargement Initial des Données (using imported functions) ---
print("Loading initial data...")
try:
    toilet_daily_data, toilet_monthly_data = get_toilet_data()
    print(f"Toilet data loaded: Daily shape={toilet_daily_data.shape}, Monthly shape={toilet_monthly_data.shape}")
except Exception as e:
    print(f"Error running get_toilet_data: {e}")
    toilet_daily_data, toilet_monthly_data = pd.DataFrame(), pd.DataFrame()

try:
    sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = get_sleep_data()
    print(f"Sleep data loaded: Daily shape={sleep_daily_data.shape}, Monthly shape={sleep_monthly_data.shape}, Failures shape={sleep_bed_failure_daily_markers.shape}")
except Exception as e:
    print(f"Error running get_sleep_data: {e}")
    sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
print("Initial data loading complete.")

# --- Initialisation de l'App Dash ---
app = Dash(__name__)
app.title = "Activity Dashboard Manager"

# --- Définition du Layout de l'App ---
app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px', 'minHeight': '100vh'}, children=[
    html.H2(app.title, style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Sélecteur d'Activité
    html.Div([
        html.Label("Select Activity Type:", style={'margin-right': '10px'}),
        dcc.Dropdown(
            id='activity-type-selector',
            options=[
                {'label': TOILET_APP_NAME, 'value': 'toilet'},
                {'label': SLEEP_APP_NAME, 'value': 'sleep'}
            ],
            value='toilet', # Default value
            clearable=False,
            style={'width': '250px', 'display': 'inline-block', 'color': '#333'}
        ),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Sélecteur d'Échelle (Année/Mois)
    html.Div([
        html.Label("Select view scale:", style={'margin-right': '10px'}),
        dcc.RadioItems(
            id='scale-selector',
            options=[
                {'label': 'Year View (Monthly)', 'value': 'year'},
                {'label': 'Month View (Daily)', 'value': 'month'}
            ],
            value='year',
            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
            inputStyle={'margin-right': '5px'}
        ),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Conteneur pour le Menu Déroulant des Mois
    html.Div(id='month-dropdown-container', children=[
        html.Label("Select Month:", style={'margin-right': '10px'}),
        dcc.Dropdown(
            id='month-dropdown',
            clearable=False,
            style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
        )
    ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}), # Initially hidden

    # Graphique
    dcc.Graph(id='activity-graph', style={'height': '60vh'})
])

# --- Callbacks ---

# Callback pour afficher/cacher le menu déroulant des mois
@app.callback(
    Output('month-dropdown-container', 'style'),
    Input('scale-selector', 'value')
)
def toggle_month_dropdown_visibility(scale):
    display = 'block' if scale == 'month' else 'none'
    return {'display': display, 'textAlign': 'center', 'marginBottom': '20px'}

# Callback pour mettre à jour les options du menu déroulant des mois
@app.callback(
    Output('month-dropdown', 'options'),
    Output('month-dropdown', 'value'),
    Input('activity-type-selector', 'value')
)
def update_month_dropdown_options(selected_activity_type):
    options = []
    value = None
    df_daily_source = pd.DataFrame() # Default to empty

    if selected_activity_type == 'toilet':
        df_daily_source = toilet_daily_data
    elif selected_activity_type == 'sleep':
        df_daily_source = sleep_daily_data

    if not df_daily_source.empty and 'year_month' in df_daily_source.columns:
        available_months = sorted(df_daily_source['year_month'].unique())
        options = [{'label': m, 'value': m} for m in available_months]
        if available_months:
            value = available_months[0]
    return options, value

# Callback principal pour mettre à jour le graphique (using imported functions)
@app.callback(
    Output('activity-graph', 'figure'),
    Input('activity-type-selector', 'value'),
    Input('scale-selector', 'value'),
    Input('month-dropdown', 'value')
)
def update_main_graph(activity_type, scale, selected_month):
    """Updates the main graph by calling the appropriate figure creation function."""
    fig = go.Figure() # Initialize empty figure

    try:
        if activity_type == 'toilet':
            # Call the imported function to create the toilet figure
            fig = create_toilet_figure(toilet_daily_data, toilet_monthly_data, scale, selected_month)
        elif activity_type == 'sleep':
            # Call the imported function to create the sleep figure
            fig = create_sleep_figure(sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers, scale, selected_month)
        else:
             fig.update_layout(title=dict(text="Select an activity type", font=dict(color=TEXT_COLOR)))

        # Ensure basic layout properties if fig was modified by create functions
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR
        )

    except Exception as e:
        print(f"Error creating figure for {activity_type}: {e}")
        fig = go.Figure() # Reset to empty figure on error
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR,
            title=dict(text=f"Error generating graph for {activity_type}", font=dict(color='red'))
        )


    # Final fallback title if no data exists in the generated figure
    if not fig.data and 'title' not in fig.layout:
         fig.update_layout(title=dict(text="Veuillez sélectionner une activité et une échelle.", font=dict(color=TEXT_COLOR)))


    return fig

# --- Exécution de l'App ---
if __name__ == '__main__':
    print("Starting Dash Manager App...")
    app.run(debug=True)
