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
    def get_toilet_data(): return pd.DataFrame(), pd.DataFrame()
    def create_toilet_figure(*args): return go.Figure().update_layout(title_text="Error loading toilet module")

try:
    from visualize_sleep_quiet_new import get_sleep_data, create_sleep_figure
    print("Successfully imported sleep functions.")
except ImportError as e:
    print(f"Error importing from visualize_sleep_quiet_new.py: {e}")
    def get_sleep_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_sleep_figure(*args): return go.Figure().update_layout(title_text="Error loading sleep module")

try:
    # Corrected import name if your file is visualize_outings_new.py
    from visualize_outing_new import get_outings_data, create_outings_figure
    print("Successfully imported outings functions.")
except ImportError as e:
    print(f"Error importing from visualize_outings_new.py: {e}") # Corrected filename in error message
    def get_outings_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_outings_figure(*args): return go.Figure().update_layout(title_text="Error loading outings module")


# --- Configuration Globale ---
# These Python variables are still useful for Plotly figure styling within create_..._figure functions
# and for fallback styles if CSS doesn't load, but primary styling is now in CSS.
TEXT_COLOR_PY = 'white'
BACKGROUND_COLOR_PY = '#111111' 

TOILET_APP_NAME = "Toilet Activity"
SLEEP_APP_NAME = "Sleep Activity"
OUTINGS_APP_NAME = "Outings Activity"

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
# Dash automatically includes CSS files from the 'assets' folder if it exists.
app = Dash(__name__)
app.title = "Activity Dashboard Manager"

# --- Définition du Layout de l'App ---
app.layout = html.Div(id="app-container", children=[ 

    html.H2(app.title, id="app-title"),

    #block des controles
    html.Div(id="control-container", children=[

            # Sélecteur d'Activité
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
                    # labelStyle and inputStyle are often kept for dcc components
                    # as they target specific sub-elements not easily reachable by general CSS.
                    labelStyle={'display': 'inline-block', 'marginRight': '22px', 'color': TEXT_COLOR_PY}, #
                    inputStyle={'marginRight': '5px'}
                    # Removed inline style from the parent Div
                ),
            ]),
       
            # Conteneur pour le Menu Déroulant des Mois
            html.Div(id='month-dropdown-container', className='selector', children=[ 

                html.Label(""), #Select Month: 
                dcc.Dropdown(
                    id='month-dropdown',
                    clearable=False,
                    className='dash-dropdown' # General class for dropdowns
                )

            ], style={'display': 'none'}), # Dynamic display style remains controlled by callback

         
    ]),

    # Graphique
    dcc.Graph(id='activity-graph') 
])

# --- Callbacks ---

# Callback pour afficher/cacher le menu déroulant des mois
@app.callback(
    Output('month-dropdown-container', 'style'),
    Input('scale-selector', 'value')
)
def toggle_month_dropdown_visibility(scale):
    """Shows/hides the month dropdown based on the selected scale."""
    # Only control the display property dynamically. Other styles are in CSS.
    if scale == 'month':
        return {'display': 'flex'} 
    else:
        return {'display': 'none'}


# Callback pour mettre à jour les options du menu déroulant des mois
@app.callback(
    Output('month-dropdown', 'options'),
    Output('month-dropdown', 'value'),
    Input('activity-type-selector', 'value')
)
def update_month_dropdown_options(selected_activity_type):
    """Updates the month dropdown options and value based on the selected activity."""
    options = []
    value = None
    df_daily_source = pd.DataFrame()

    if selected_activity_type == 'toilet':
        df_daily_source = toilet_daily_data
    elif selected_activity_type == 'sleep':
        df_daily_source = sleep_daily_data
    elif selected_activity_type == 'outings':
        df_daily_source = outings_daily_data

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
    fig = go.Figure()

    try:
        if activity_type == 'toilet':
            fig = create_toilet_figure(toilet_daily_data, toilet_monthly_data, scale, selected_month)
        elif activity_type == 'sleep':
            fig = create_sleep_figure(sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers, scale, selected_month)
        elif activity_type == 'outings':
            fig = create_outings_figure(outings_daily_data, outings_monthly_data, outings_door_failure_daily_markers, scale, selected_month)
        else:
             fig.update_layout(title=dict(text="Select an activity type", font=dict(color=TEXT_COLOR_PY)))

        # The create_..._figure functions should handle their specific Plotly templates and colors.
        # We ensure the general background from CSS is respected by Plotly if not overridden.
        # fig.update_layout( # This might override specific styling from create_..._figure if not careful
        #     template='plotly_dark', # This is good, but create_..._figure should also set it
        #     paper_bgcolor=BACKGROUND_COLOR_PY, # Set by create_..._figure
        #     plot_bgcolor=BACKGROUND_COLOR_PY   # Set by create_..._figure
        # )

    except Exception as e:
        print(f"Error creating figure for {activity_type}: {e}")
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=BACKGROUND_COLOR_PY,
            plot_bgcolor=BACKGROUND_COLOR_PY,
            font=dict(color=TEXT_COLOR_PY),
            title=dict(text=f"Error generating graph for {activity_type}", font=dict(color='red'))
        )

    if not fig.data and (not fig.layout or not fig.layout.title or not fig.layout.title.text):
         fig.update_layout(
             template='plotly_dark',
             paper_bgcolor=BACKGROUND_COLOR_PY,
             plot_bgcolor=BACKGROUND_COLOR_PY,
             font=dict(color=TEXT_COLOR_PY),
             title=dict(text="Veuillez sélectionner une activité et une échelle.", font=dict(color=TEXT_COLOR_PY))
        )
    return fig

# --- Exécution de l'App ---
if __name__ == '__main__':
    print("Starting Dash Manager App...")
    app.run(debug=True)
