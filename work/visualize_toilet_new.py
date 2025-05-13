import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np # Import numpy for potential use

# --- Configuration ---
TOILET_LOG_FILE = 'rule-toilet.csv'
APP_TITLE = "Toilet Activity Viewer"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' 
DATA2_COLOR = '#F14864'
DATA3_COLOR = '#EB9636'
DATAMONTH_COLOR = '#36A0EB'
#43D37B # green

# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5) 
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3) 

TITLE_X = 0.06
TITLE_Y = 0.92

# --- Data Loading and Processing Function ---
def get_toilet_data():
    """Loads and preprocesses toilet activity data, returning daily and monthly aggregations."""
    try:
        activity = pd.read_csv(TOILET_LOG_FILE, delimiter=';', decimal=",",
                               names=["date", "annotation", "activity_count", "duration"],
                               parse_dates=["date"], index_col="date")
        # Conversion de la durée de secondes en minutes
        activity['duration_min'] = activity['duration'] / 60
    except FileNotFoundError:
        print(f"Error: '{TOILET_LOG_FILE}' not found.")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                index=pd.to_datetime([]))
        activity.index.name = 'date'
    except Exception as e:
        print(f"Error loading {TOILET_LOG_FILE}: {e}")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                index=pd.to_datetime([]))
        activity.index.name = 'date'

    # Daily Aggregation
    if not activity.empty:
        activity_daily = activity.resample('D').agg(
            activity_count_sum=('activity_count', 'sum'),
            duration_min_mean=('duration_min', 'mean'), # Utilise duration_min
            duration_min_sem=('duration_min', 'sem')    # Utilise duration_min
        ).reset_index()
        activity_daily['year_month'] = activity_daily['date'].dt.to_period('M').astype(str)
    else:
        activity_daily = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_mean', 'duration_min_sem', 'year_month'])

    # Monthly Aggregation
    if not activity.empty:
        activity_monthly = activity.resample('ME').agg(
            activity_count_sum=('activity_count', 'sum'),
            duration_min_mean=('duration_min', 'mean'), # Utilise duration_min
            duration_min_sem=('duration_min', 'sem')    # Utilise duration_min
        ).reset_index()
        if pd.api.types.is_datetime64_any_dtype(activity_monthly['date']):
            activity_monthly['days_in_month'] = activity_monthly['date'].dt.daysinmonth
            activity_monthly['activity_count_mean_daily'] = np.where(
                activity_monthly['days_in_month'] > 0,
                activity_monthly['activity_count_sum'] / activity_monthly['days_in_month'],
                0
            )
        else:
            activity_monthly['days_in_month'] = 0
            activity_monthly['activity_count_mean_daily'] = 0
        # Ajout du format MM/YY pour l'affichage des mois
        activity_monthly['month_label'] = activity_monthly['date'].dt.strftime('%m/%y')
    else:
       activity_monthly = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_mean', 'duration_min_sem', 'days_in_month', 'activity_count_mean_daily', 'month_label'])

    return activity_daily, activity_monthly

# --- Figure Creation Function ---
def create_toilet_figure(daily_data, monthly_data, scale, selected_month):
    
    """Creates the Plotly figure for toilet activity based on inputs."""
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
        legend=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        margin = MARGIN_CHART
        
    )

    if scale == 'year':
        df = monthly_data
        if not df.empty:
            fig.add_trace(go.Bar(x=df['month_label'], y=df['duration_min_mean'], name="Durée moyenne (min)", error_y=dict(type='data', array=df['duration_min_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df['month_label'], y=df['activity_count_sum'], name="Passages totaux", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR)))
            if 'activity_count_mean_daily' in df.columns:
                fig.add_trace(go.Scatter(x=df['month_label'], y=df['activity_count_mean_daily'], name="Passages moyens/jour", yaxis='y2', mode='lines+markers', line=dict(color=DATA3_COLOR)))
            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité mensuelle", font=dict(color=TEXT_COLOR),x = TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Durée moyenne (min)", font=dict(color=DATA1_COLOR)), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Nombre de Passages", font=dict(color=DATA3_COLOR)), overlaying='y', side='right', tickfont=dict(color=DATA3_COLOR)),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly data available", font=dict(color=TEXT_COLOR)))

    elif scale == 'month' and selected_month:
        df = daily_data[daily_data['year_month'] == selected_month]
        if not df.empty:
            fig.add_trace(go.Bar(x=df['date'], y=df['activity_count_sum'], name="Passages par jour", marker_color=DATAMONTH_COLOR))
            fig.update_layout(
                title=dict(text=f"Vue Journalière : {selected_month}", font=dict(color=TEXT_COLOR), x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Nombre de passages", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR)),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Monthly View: No data available for {selected_month}", font=dict(color=TEXT_COLOR)))
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month", font=dict(color=TEXT_COLOR)))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher", font=dict(color=TEXT_COLOR)))

    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    # Load data when running standalone
    activity_daily_data, activity_monthly_data = get_toilet_data()
    available_months = sorted(activity_daily_data['year_month'].unique()) if not activity_daily_data.empty else []

    # Initialize the app
    app = Dash(__name__)
    app.title = APP_TITLE

    # Define the layout
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='scale-selector',
                options=[{'label': 'Year View (Monthly)', 'value': 'year'}, {'label': 'Month View (Daily)', 'value': 'month'}],
                value='year',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                inputStyle={'marginRight': '5px'}
            ),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div(id='month-dropdown-container', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='activity-graph', style={'height': '60vh'})
    ])

    # Callback to show/hide month dropdown
    @app.callback(
        Output('month-dropdown-container', 'style'),
        Input('scale-selector', 'value')
    )
    def toggle_month_dropdown(scale):
        display = 'block' if scale == 'month' else 'none'
        return {'display': display, 'textAlign': 'center', 'marginBottom': '20px'}

    # Callback to update graph
    @app.callback(
        Output('activity-graph', 'figure'),
        Input('scale-selector', 'value'),
        Input('month-dropdown', 'value')
    )
    def update_graph_standalone(scale, selected_month):
        # Call the figure creation function using pre-loaded data
        return create_toilet_figure(activity_daily_data, activity_monthly_data, scale, selected_month)

    # Run the app
    app.run(debug=True)
