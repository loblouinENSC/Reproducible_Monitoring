import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import numpy as np

# --- Configuration ---
SLEEP_LOG_FILE = 'rule-sleep_quiet.csv'
BED_FAILURE_FILE = 'days-bed_failure.csv'
APP_TITLE = "Sleeping Activity Dashboard"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'

# --- Data Loading and Processing Function ---
def get_sleep_data():
    """Loads and preprocesses sleep and bed failure data."""
    try:
        sleep = pd.read_csv(SLEEP_LOG_FILE, delimiter=';', decimal=",",
                               names=["date", "annotation", "sleep_count", "duration"],
                               parse_dates=["date"], index_col="date")
        sleep['durationHr'] = sleep['duration'] / 3600
        sleep = sleep[['durationHr']]
    except FileNotFoundError:
        print(f"Error: '{SLEEP_LOG_FILE}' not found.")
        sleep = pd.DataFrame(columns=["durationHr"], index=pd.to_datetime([]))
        sleep.index.name = 'date'
    except Exception as e:
        print(f"Error loading {SLEEP_LOG_FILE}: {e}")
        sleep = pd.DataFrame(columns=["durationHr"], index=pd.to_datetime([]))
        sleep.index.name = 'date'

    try:
        bed_failure = pd.read_csv(BED_FAILURE_FILE, delimiter=';', decimal=",",
                                     names=["date", "bed_failure"],
                                     parse_dates=["date"], index_col="date")
        bed_failure['bed_failure'] = pd.to_numeric(bed_failure['bed_failure'], errors='coerce').fillna(0)
        bed_failure_daily_markers = bed_failure[bed_failure['bed_failure'] > 0].copy().reset_index()
    except FileNotFoundError:
        print(f"Error: '{BED_FAILURE_FILE}' not found.")
        bed_failure = pd.DataFrame(columns=["bed_failure"], index=pd.to_datetime([]))
        bed_failure.index.name = 'date'
        bed_failure_daily_markers = pd.DataFrame(columns=['date', 'bed_failure'])
    except Exception as e:
        print(f"Error loading {BED_FAILURE_FILE}: {e}")
        bed_failure = pd.DataFrame(columns=["bed_failure"], index=pd.to_datetime([]))
        bed_failure.index.name = 'date'
        bed_failure_daily_markers = pd.DataFrame(columns=['date', 'bed_failure'])

    # Daily Aggregation (Sleep)
    if not sleep.empty:
        sleep_daily = sleep.resample('D').agg(duration_sum=('durationHr', 'sum')).reset_index()
        sleep_daily['year_month'] = sleep_daily['date'].dt.to_period('M').astype(str)
    else:
        sleep_daily = pd.DataFrame(columns=['date', 'duration_sum', 'year_month'])

    # Monthly Aggregation (Sleep & Bed Failure)
    if not sleep.empty:
        sleep_monthly_agg = sleep.resample('ME').agg(
            duration_mean=('durationHr', 'mean'),
            duration_sem=('durationHr', 'sem')
        )
    else:
        sleep_monthly_agg = pd.DataFrame(columns=['duration_mean', 'duration_sem'], index=pd.to_datetime([]))
        sleep_monthly_agg.index.name = 'date'

    if not bed_failure.empty:
        bed_failure_monthly_agg = bed_failure.fillna(0).resample('ME').agg(
            bed_failure_sum=('bed_failure', 'sum')
         )
    else:
        bed_failure_monthly_agg = pd.DataFrame(columns=['bed_failure_sum'], index=pd.to_datetime([]))
        bed_failure_monthly_agg.index.name = 'date'

    sleep_monthly = pd.merge(sleep_monthly_agg, bed_failure_monthly_agg, left_index=True, right_index=True, how='outer')
    if not sleep_monthly.empty:
        start_date, end_date = sleep_monthly.index.min(), sleep_monthly.index.max()
        if pd.notna(start_date) and pd.notna(end_date):
            full_idx = pd.date_range(start=start_date, end=end_date, freq='ME')
            sleep_monthly = sleep_monthly.reindex(full_idx)
        sleep_monthly['duration_mean'] = sleep_monthly['duration_mean'].fillna(np.nan)
        sleep_monthly['duration_sem'] = sleep_monthly['duration_sem'].fillna(0)
        sleep_monthly['bed_failure_sum'] = sleep_monthly['bed_failure_sum'].fillna(0)
    sleep_monthly = sleep_monthly.reset_index().rename(columns={'index': 'date'})

    # Add year_month to daily markers for filtering in the figure function
    if not bed_failure_daily_markers.empty and 'date' in bed_failure_daily_markers.columns:
         if pd.api.types.is_datetime64_any_dtype(bed_failure_daily_markers['date']):
             bed_failure_daily_markers['year_month'] = bed_failure_daily_markers['date'].dt.to_period('M').astype(str)
         else: # Attempt conversion if not already datetime
             bed_failure_daily_markers['date'] = pd.to_datetime(bed_failure_daily_markers['date'], errors='coerce')
             bed_failure_daily_markers.dropna(subset=['date'], inplace=True)
             if not bed_failure_daily_markers.empty:
                 bed_failure_daily_markers['year_month'] = bed_failure_daily_markers['date'].dt.to_period('M').astype(str)

    return sleep_daily, sleep_monthly, bed_failure_daily_markers

# --- Figure Creation Function ---
def create_sleep_figure(daily_data, monthly_data, daily_failure_markers, scale, selected_month):
    """Creates the Plotly figure for sleep activity based on inputs."""
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
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)')
    )

    if scale == 'year':
        df = monthly_data
        if not df.empty:
            fig.add_trace(go.Bar(x=df['date'], y=df['duration_mean'], name="Durée moyenne sommeil (h)", error_y=dict(type='data', array=df['duration_sem']), marker_color='blue'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['bed_failure_sum'], name="Jours échec lit (total)", yaxis='y2', mode='lines+markers', line=dict(color='red')))
            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité de Sommeil Mensuelle", font=dict(color=TEXT_COLOR)),
                xaxis=dict(title=dict(text="Mois", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Durée moyenne sommeil (h)", font=dict(color='blue')), tickfont=dict(color='blue')),
                yaxis2=dict(title=dict(text="Jours échec lit", font=dict(color='red')), overlaying='y', side='right', tickfont=dict(color='red'), showgrid=False, range=[0, df['bed_failure_sum'].max() * 1.1 if pd.notna(df['bed_failure_sum'].max()) and df['bed_failure_sum'].max() > 0 else 5]),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly data available", font=dict(color=TEXT_COLOR)))

    elif scale == 'month' and selected_month:
        df_daily_s = daily_data[daily_data['year_month'] == selected_month]
        # Filter daily failure markers for the selected month
        df_daily_f = pd.DataFrame() # Initialize empty
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns:
             df_daily_f = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]

        if not df_daily_s.empty:
            fig.add_trace(go.Bar(x=df_daily_s['date'], y=df_daily_s['duration_sum'], name="Durée sommeil (h)", marker_color='green'))
            if not df_daily_f.empty:
                fig.add_trace(go.Scatter(x=df_daily_f['date'], y=[0.1] * len(df_daily_f), name="Échec lit", mode='markers', marker=dict(color='red', size=10, symbol='x')))
            fig.update_layout(
                title=dict(text=f"Vue Journalière : {selected_month}", font=dict(color=TEXT_COLOR)),
                xaxis=dict(title=dict(text="Jour", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Durée sommeil (h)", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR))
            )
        else:
            fig.update_layout(title=dict(text=f"Monthly View: No sleep data available for {selected_month}", font=dict(color=TEXT_COLOR)))
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month", font=dict(color=TEXT_COLOR)))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher", font=dict(color=TEXT_COLOR)))

    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    # Load data when running standalone
    sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = get_sleep_data()
    available_months = sorted(sleep_daily_data['year_month'].unique()) if not sleep_daily_data.empty else []

    # Initialize the app
    app = Dash(__name__)
    app.title = APP_TITLE

    # Define the layout
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
         html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
         html.Div([
             html.Label("Select view scale:", style={'margin-right': '10px'}),
             dcc.RadioItems(
                 id='scale-selector-sleep',
                 options=[{'label': 'Year View (Monthly)', 'value': 'year'}, {'label': 'Month View (Daily)', 'value': 'month'}],
                 value='year',
                 labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                 inputStyle={'margin-right': '5px'}
             ),
         ], style={'marginBottom': '20px', 'textAlign': 'center'}),
         html.Div(id='month-dropdown-container-sleep', children=[
             html.Label("Select Month:", style={'margin-right': '10px'}),
             dcc.Dropdown(
                 id='month-dropdown-sleep',
                 options=[{'label': m, 'value': m} for m in available_months],
                 value=available_months[0] if len(available_months) > 0 else None,
                 clearable=False,
                 style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
             )
         ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),
         dcc.Graph(id='sleep-activity-graph', style={'height': '60vh'})
    ])

    # Callback to show/hide month dropdown
    @app.callback(
        Output('month-dropdown-container-sleep', 'style'),
        Input('scale-selector-sleep', 'value')
    )
    def toggle_month_dropdown_sleep(scale):
        display = 'block' if scale == 'month' else 'none'
        return {'display': display, 'textAlign': 'center', 'marginBottom': '20px'}

    # Callback to update graph
    @app.callback(
        Output('sleep-activity-graph', 'figure'),
        Input('scale-selector-sleep', 'value'),
        Input('month-dropdown-sleep', 'value')
    )
    def update_graph_standalone(scale, selected_month):
        # Call the figure creation function using pre-loaded data
        return create_sleep_figure(sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers, scale, selected_month)

    # Run the app
    app.run(debug=True)
